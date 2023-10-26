import numpy as np
import pandas as pd
from ipfx import feature_extractor
from scipy.stats import *
from scipy import interpolate
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from brian2 import pF, pA, nS, mV, NeuronGroup, pamp, run, second, StateMonitor, ms, TimedArray, size, nan, array, reshape, \
    shape, volt, siemens, amp, farad, ohm, Gohm
from brian2 import *
from loadNWB import loadNWB
try:
    ### these are some libraries for spike train assesment not needed if you are not calling spike dist
    from elephant.spike_train_dissimilarity import victor_purpura_dist, van_rossum_dist
    from neo.core import SpikeTrain
    import quantities as pq 
    import matplotlib.pyplot as plt
except:
    print('Spike distance lib import failed')


#from smoothn import *
from brian2 import plot
import matplotlib.pyplot as plt


def detect_spike_times(dataX, dataY, dataC, sweeps=None, dvdt=20, swidth=10, speak=-10, lower=None, upper=None):
    # requires IPFX (allen institute). Modified version by smestern
    # install using pip install git+https://github.com/smestern/ipfx.git Not on git yet
    # works with abf and nwb
    swidth /= 1000
    if sweeps is None:
        sweepList = np.arange(dataX.shape[0])
    else:
        sweepList = np.asarray(sweeps)

    if lower is None:
        lower = 0
    if upper is None:
        upper = dataX[0, -1]

    spikedect = feature_extractor.SpikeFeatureExtractor(filter=0, dv_cutoff=dvdt, max_interval=swidth, min_peak=speak, start=lower, end=upper)
    spike_list = []
    for sweep in sweepList:
        sweepX = dataX[sweep, :]
        sweepY = dataY[sweep, :]
        sweepC = dataC[sweep, :]
        try:
            spikes_in_sweep = spikedect.process(sweepX, sweepY, sweepC)  ##returns a dataframe
        except:
            spikes_in_sweep = pd.DataFrame()
        if spikes_in_sweep.empty == True:
            spike_list.append([])
        else:
            spike_ind = spikes_in_sweep['peak_t'].to_numpy()
            spike_list.append(spike_ind)
            
            
    return spike_list


def spikeIndices(V):
    # smooth the first difference of V; peaks should be clear and smooth
    dV = np.diff(V)
    dVsm, _, _, _ = smoothn(dV)
    
    # define spikes at indices where smoothed dV exceeds twice the standard deviation
    sigma = np.std(dVsm)
    spkIdxs, _ = find_peaks(dVsm, height=2*sigma)
    
    return spkIdxs


def compute_threshold(dataX, dataY, dataC, sweeps, dvdt=20):
    # requires IPFX (allen institute). Modified version by smestern
    # install using pip install git+https://github.com/smestern/ipfx.git Not on git yet
    # works with abf and nwb
    if sweeps is None:
        sweepList = np.arange(dataX.shape[0])
    else:
        sweepList = np.asarray(sweeps)
    spikedect = feature_extractor.SpikeFeatureExtractor(filter=0, dv_cutoff=dvdt)
    threshold_list = []
    for sweep in sweepList:
        sweepX = dataX[sweep, :]
        sweepY = dataY[sweep, :]
        sweepC = dataC[sweep, :]
        spikes_in_sweep = spikedect.process(sweepX, sweepY, sweepC)  ##returns a dataframe
        if spikes_in_sweep.empty == False:
            thres_V = spikes_in_sweep['threshold_v'].to_numpy()
            threshold_list.append(thres_V)
    try:
        return np.nanmean(threshold_list[0])
    except:
        return 0


def compute_dt(dataX):
    dt = dataX[0, 1] - dataX[0, 0]
    dt = dt * 1000  # ms
    return dt

def compute_steady_hyp(dataY, dataC, ind=[0,1]):
    stim_index = find_stim_changes(dataC[0,:])
    if len(stim_index) < 2:
        mean_steady= np.nanmean(dataY[:,:])
    else:
        mean_steady= np.nanmean(dataY[:, stim_index[ind[0]]:stim_index[ind[1]]])
    return mean_steady

def compute_rmp(dataY, dataC):
    try:
        deflection = np.nonzero(dataC[0, :])[0][0] - 1
    except:
        deflection = -1
    rmp1 = np.nanmean(dataY[:, :deflection])
    rmp2 = mode(dataY[:, :deflection], axis=None)[0][0]

    return rmp1

def find_stim_changes(dataI):
    diff_I = np.diff(dataI)
    infl = np.nonzero(diff_I)[0]
    
    '''
    dI = np.diff(np.hstack((0, dataI, 0))
    '''
    return infl

def find_decline_fi(spikes):
    spike_count = [len(x) for x in spikes]
    decline = np.where(np.diff(spike_count)<-1,1, 0)
    inflection = np.nonzero(decline)[0]
    if len(inflection)>0:
        sweep_upper = inflection[0]  
    else:
        sweep_upper = None
    return sweep_upper

def find_downward(dataI):
    diff_I = np.diff(dataI)
    downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
    return downwardinfl

def exp_decay_1p(t, a, b1, alphaFast):
    return (a + b1*(1-np.exp(-t/alphaFast)))

def exp_decay_factor(dataT,dataV,dataI, time_aft=50, plot=False, sag=True):
    try:
        time_aft = time_aft / 100
        if time_aft > 1:
            time_aft = 1
        if sag:
            diff_I = np.diff(dataI)
            downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
            
            end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
            upperC = np.amax(dataV[downwardinfl:end_index])
            lowerC = np.amin(dataV[downwardinfl:end_index])
            minpoint = np.argmin(dataV[downwardinfl:end_index])
            end_index = downwardinfl + int(.99 * minpoint)
            downwardinfl = downwardinfl #+ int(.10 * minpoint)
        else:
            diff_I = np.diff(dataI)
            downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
            end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
            
            upperC = np.amax(dataV[downwardinfl:end_index])
            lowerC = np.amin(dataV[downwardinfl:end_index])
        diff = np.abs(upperC - lowerC)
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        curve, pcov_1p = curve_fit(exp_decay_1p, t1, dataV[downwardinfl:end_index]/1000, maxfev=500000, bounds=([(upperC-0.5)/1000, -np.inf, 0], [(upperC+0.5)/1000, np.inf, np.inf]), xtol=None, verbose=1)
        tau = curve[2]
        if plot:
            plt.figure(2)
            plt.clf()
            plt.plot(t1, dataV[downwardinfl:end_index]/1000, label='Data')
            plt.plot(t1, exp_decay_1p(t1, *curve), label='1 phase fit')
            
            plt.legend()
            
            plt.pause(3)
        return tau
    except:
        return 0

def compute_sag(dataT,dataV,dataI, time_aft=50):
    min_max = [np.argmin, np.argmax]
    find = 0
    time_aft = time_aft / 100
    if time_aft > 1:
        time_aft = 1   
    diff_I = np.diff(dataI)
    upwardinfl = np.nonzero(np.where(diff_I>0, diff_I, 0))[0][0]
    downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
    if upwardinfl < downwardinfl: #if its depolarizing then swap them
        temp = downwardinfl
    find = 1
    downwardinfl = upwardinfl
    upwardinfl = temp
    dt = dataT[1] - dataT[0] #in s
    end_index = upwardinfl - int(0.100/dt)
    end_index2 = upwardinfl - int((upwardinfl - downwardinfl) * time_aft)
    if end_index<downwardinfl:
        end_index = upwardinfl - 5
    vm = np.nanmean(dataV[end_index:upwardinfl])
    min_point = downwardinfl + min_max[find](dataV[downwardinfl:end_index2])
    avg_min = np.nanmean(dataV[min_point])
    sag_diff = avg_min - vm
    return sag_diff, 

def membrane_resistance(dataT,dataV,dataI):
    try:
        diff_I = np.diff(dataI)
        downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
        end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl)/2)
        
        upperC = np.mean(dataV[:downwardinfl-100])
        lowerC = np.mean(dataV[downwardinfl+100:end_index-100])
        diff = -1 * np.abs(upperC - lowerC)
        I_lower = dataI[downwardinfl+1]
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        #v = IR
        #r = v/I
        v_ = diff / 1000 # in mv -> V
        I_ = I_lower / 1000000000000 #in pA -> A
        r = v_/I_

        return r #in ohms
    except: 
        return np.nan

def membrane_resistance_subt(dataT, dataV,dataI):
    resp_data = []
    stim_data = []
    for i, sweep in enumerate(dataV):
        abs_min, resp = compute_sag(dataT[i,:], sweep, dataI[i,:])
        ind = find_stim_changes(dataI[i, :])
        baseline = np.mean(sweep[:ind[0]])
        stim = dataI[i,ind[0] + 1]
        stim_data.append(stim)
        resp_data.append((resp+abs_min) - baseline)
    resp_data = np.array(resp_data) * mV
    stim_data = np.array(stim_data) * pA
    res = linregress(stim_data / amp, resp_data / volt)
    resist = res.slope * ohm
    return resist / Gohm


def mem_cap(resist, tau_1p):
    #tau = RC
    #C = R/tau
    
    C_1p = tau_1p / resist
    return C_1p ##In farads?

def plot_adex_state(adex_state_monitor):
    """
    Visualizes the state variables: w-t, v-t and phase-plane w-v
    from https://github.com/EPFL-LCN/neuronaldynamics-exercises/
    Args:
        adex_state_monitor (StateMonitor): States of "v" and "w"

    """
    import matplotlib.pyplot as plt
    plt.figure(num=12, figsize=(10,10))
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(adex_state_monitor.t / ms, adex_state_monitor.v[0] / mV, lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("u [mV]")
    plt.title("Membrane potential")
    plt.subplot(2, 2, 2)
    plt.plot(adex_state_monitor.v[0] / mV, adex_state_monitor.w[0] / pA, lw=2)
    plt.xlabel("u [mV]")
    plt.ylabel("w [pAmp]")
    plt.title("Phase plane representation")
    plt.subplot(2, 2, 3)
    plt.plot(adex_state_monitor.t / ms, adex_state_monitor.w[0] / pA, lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("w [pAmp]")
    plt.title("Adaptation current")

def compute_sse(y, yhat):
    mse = np.sum(np.square(y - yhat))
    return mse


def compute_mse(y, yhat):
    mse = np.mean(np.square(y - yhat))
    return mse

def compute_mlse(y, yhat):
    mse = np.mean(np.square(np.log10(y+1) - np.log10(yhat+1)))
    return mse

def compute_se(y, yhat):
    se = np.square(y - yhat)
    return se

def compute_ae(y, yhat):
    ae = np.abs(y - yhat)
    return ae

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def equal_array_size_1d(array1, array2, method='append', append_val=0):
    ar1_size = array1.shape[0]
    ar2_size = array2.shape[0]
    if ar1_size == ar2_size:
        pass
    elif method == 'append':
        if ar1_size > ar2_size:
            array2 = np.hstack((array2, np.full(ar1_size - ar2_size, append_val)))
        elif ar2_size > ar1_size:
            array1 = np.hstack((array1, np.full(ar2_size - ar1_size, append_val)))
    elif method == 'trunc':
        if ar1_size > ar2_size:
            array1 = array1[:ar2_size]
        elif ar2_size > ar1_size:
            array2 = array2[:ar1_size]
    elif method == 'interp':
        if ar1_size > ar2_size:
            interp = interpolate.interp1d(np.linspace(1,ar2_size-1, ar2_size), array2, bounds_error=False, fill_value='extrapolate')
            new_x = np.linspace(ar2_size, ar1_size, (ar1_size - ar2_size))
            array2 = np.hstack((array2, interp(new_x)))
        elif ar2_size > ar1_size:
            interp = interpolate.interp1d(np.linspace(1,ar1_size-1, ar1_size), array1, bounds_error=False, fill_value='extrapolate')
            new_x = np.linspace(ar1_size, ar2_size, (ar2_size - ar1_size))
            array2 = np.hstack((array1, interp(new_x)))
    return array1, array2


def compute_spike_dist(y, yhat):
    '''
    Computes the distance between the two spike trains
    takes arrays of spike times in seconds
    '''
    #y, yhat = equal_array_size_1d(y, yhat, 'append')

    
    train1 = SpikeTrain(y*pq.s, t_stop=6*pq.s)
    train2 = SpikeTrain(yhat*pq.s, t_stop=6*pq.s)
    
    dist = van_rossum_dist([train1, train2], tau=50*pq.ms)  
    ## Update later to compute spike distance using van rossum dist
    r_dist = dist[0,1] #returns squareform so just 
    return r_dist

def compute_spike_dist_euc(y, yhat):
    '''
    Computes the distance between the two spike trains
    takes arrays of spike times in seconds
    '''
    y, yhat = equal_array_size_1d(y, yhat, 'append', append_val=0)
    if len(y) < 1 and len(yhat) < 1:
        dist = 999
    else:
        dist = distance.euclidean(y, yhat)  
    
    r_dist = dist
    return r_dist


def compute_corr(y, yhat):

    y, yhat = equal_array_size_1d(y, yhat, 'append')
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    yhat = np.nan_to_num(yhat, nan=0, posinf=0, neginf=0)
    try:
        corr_coef = pearsonr(y, yhat)
    except:
        corr_coef = 0
    return np.amax(corr_coef)

def replace_nan(a):
    temp = a.copy()
    temp[np.isnan(a)] = np.nanmax(a)
    return temp

def drop_rand_rows(a, num):
    rows = a.shape[0]-1
    rows_to_drop = np.random.rarandint(0, rows, num)
    a = np.delete(a,rows_to_drop,axis=0)
    return a

def compute_distro_mode(x, bin=20, wrange=False):
    if wrange:
        bins = np.arange(np.amin(x)-bin, np.amax(x)+bin, bin)
    else:
        if np.amax(x)<0:
            bins = np.arange(np.amax(x)-bin,0, bin)
        else:
            bins = np.arange(0, np.amax(x)+bin, bin)
    hist, bins = np.histogram(x, bins=bins)
    return bins[np.argmax(hist)]



def compute_corr_minus(y, yhat):

    y, yhat = equal_array_size_1d(y, yhat, 'append')
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    yhat = np.nan_to_num(yhat, nan=0, posinf=0, neginf=0)
    try:
        corr_coef = 1 - np.amax(pearsonr(y, yhat))
    except:
        corr_coef = 1
    return corr_coef

def compute_FI(spkind, dt, dataC):
    isi = [ dt*np.diff(x) for x in spkind ]
    f = [ np.reciprocal(x) for x in isi ]
    i = []
    for ii in range(len(dataC)):
        tmp = dataC[ii]
        tmp1 = spkind[ii][:-1]
        i.append(tmp[tmp1])
    return f, i, isi
    
def compute_min_stim(dataY, dataX, strt, end):
    #find the strt, end
    index_strt = np.argmin(np.abs(dataX - strt))
    index_end = np.argmin(np.abs(dataX - end))
    #Find the min
    amin = np.amin(dataY[index_strt:index_end])
    return amin

def compute_FI_curve(spike_times, time, bin=20):
    FI_full = []
    isi=[]
    for r in spike_times:
        if len(r) > 0:
            FI_full.append(len(r))
            if len(r) > 1:
                isi_row = np.diff(r)
                isi.append(np.nanmean(isi_row*1000))
            else:
                isi.append(0)
        else:
            FI_full.append(0)
            isi.append(0) 
    return (np.hstack(FI_full) /time), np.hstack(isi)
            
def add_spikes_to_voltage(spike_times,voltmonitor, peak=33, index=0):
    if len(spike_times) > 0:
            trace_round = np.around(voltmonitor.t/ms, decimals=0)
            spikes_round = np.around(spike_times, decimals=0)
            spike_idx = np.isin(trace_round, spikes_round)
            traces_v =  voltmonitor[index].v/mV
            traces_v[spike_idx] = peak
    else:
            traces_v =  voltmonitor[index].v/mV
    return traces_v
             

def sweepwise_qc(x, y, c):
    #remove sweeps with a voltage value outside a range of -90, -40
    sweep_wise_mode = np.apply_along_axis(compute_distro_mode, 1, y, bin=5)
    arg = np.where(np.logical_and(sweep_wise_mode>=-90, sweep_wise_mode<=200))[0]
    x = x[arg, :]
    y = y[arg,:]
    c = c[arg, :]
    return x, y, c



def plot_trace(param_dict, model):
    ''' 
    Plots the trace (for debugging purposes)
    '''
    realX, realY = model.realX, model.realY
    figure(figsize=(10,10), num=15)
    clf()
    model.set_params(param_dict)
    model.set_params({'N': 1})
    for x in [*model.subthresholdSweep, model.spikeSweep[-1]]:
        spikes, traces = model.run_current_sweep(x)
        plot(realX[x,:], traces.v[0] /mV, label="Sim sweep {x}", c='r', alpha=0.5, zorder=9999)
        plot(realX[x,:], realY[x,:], label=f"Real Sweep {x}", c='k')
        if len(spikes.t) > 0:
            scatter(spikes.t, np.full(spikes.t.shape[0], 60) ,label="Sim spike times", marker='x')
    
    
    return



def plot_IF(param_dict, model):
    ''' 
    Plots the I(current)-F(frequency of spikes) for the model
    '''
    realX, realY, realC = model.realX, model.realY, model.realC
    figure(figsize=(10,10), num=13)
    clf()
    subplot(1,2,1)
    
    model.set_params(param_dict)
    model.set_params({'N':1})
    realspikes = model._detect_real_spikes()
    real_spike_curve,real_ISI = compute_FI_curve(realspikes, model._run_time)
    
    simspikes,sim_ISI = model.build_FI_curve()
    simspikes = simspikes[0]
    mse = compute_mse(np.asarray(real_spike_curve),np.hstack(simspikes))
    plot(np.arange(simspikes.shape[0]), simspikes, label=f"Sim FI")
    plot(np.arange(simspikes.shape[0]), real_spike_curve, label=f"Real FI")
    
    legend()
    subplot(1,2,2)
    plot(np.arange(real_ISI.shape[0]), sim_ISI[0, :], label=f"Sim FI")
    plot(np.arange(real_ISI.shape[0]), real_ISI, label=f"Real FI")
    
    legend()

def model_feature_curve(model):
    real_fi, real_isi = compute_FI_curve(model.spike_times, model._run_time) #Compute the real FI curve
    real_fi = np.hstack((real_fi, real_isi))
    real_rmp = compute_rmp(model.realY, model.realC)
    real_min = []
    real_subt = []
    for x in  model.subthresholdSweep :
        temp = compute_steady_hyp(model.realY[x, :].reshape(1,-1), model.realC[x, :].reshape(1,-1))
        temp_min = compute_min_stim(model.realY[x, :], model.realX[x,:], strt=0.62, end=1.2)
        real_subt.append(temp)
        real_min.append(temp_min)
    real_subt = np.array(real_subt)        
    
    real_min = np.hstack(real_min)
    
    np_o = np.hstack((real_fi, real_rmp, real_subt, real_min))
    
    return np_o