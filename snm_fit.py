'''
Single neuron param fit
Rough coding space of several (optimizer, random, grid search) methods for optimizing
params. Can be called from the command line:

'''
import argparse
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO) #trying to log brian2 errors
import multiprocessing
import os
import time
import warnings
from pickle import dump, load

import pandas as pd
import torch
from brian2 import *
from joblib import dump, load
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi
from sbi.inference.base import infer
from scipy import stats

from b2_model.brian2_model import brian2_model
from b2_model.error import weightedErrorMetric, zErrorMetric
from b2_model.optimizer import CustomPortfolio, snmOptimizer
from loadNWB import *
from utils import *

warnings.filterwarnings("ignore")

#to allow parallel processing
prefs.codegen.target = 'cython'  # weave is not multiprocess-safe!
prefs.codegen.runtime.cython.multiprocess_safe = False
default_dtype = torch.float32

# === Global Settings ===
rounds = 3
batch_size = 15000
N = batch_size
batches = rounds

# === MAIN FUNCTION ===

def run_optimizer(file, optimizer_settings, optimizer='ng', rounds_=500, batch_size_=500, sweep_upper_cut=None):
    ''' Runs the optimizer for a given file, using user specfied params and rounds

    Takes:
    file (str) : A file path to a NWB file. To be fit to  
    (optional)  
    optimizer (str) : the optimizer protocol to use, either 'ng' or 'snpe'  
    rounds_ (int) : the number of rounds to run  
    batch_size_ (int) : the number of neuron-param combos to try in parallel  

    Returns:
    results : the best fit params for the cell  
     '''
    global rounds
    global batch_size
    rounds=rounds_ #Set the global settings to the user passed in params
    batch_size=batch_size_
    model = load_data_and_model(file, sweep_upper_cut=sweep_upper_cut) #load the nwb and model
    cell_id = file.split("\\")[-1].split(".")[0] #grab the cell id by cutting around the file path
    print(f"== Loaded cell {cell_id} for fitting ==")
    if optimizer == 'skopt' or optimizer=='ng':
        results = _opt(model, optimizer_settings, optimizer=optimizer, id=cell_id)
    elif optimizer == 'snpe'  or optimizer=='sbi': 
        results = SNPE_OPT(model, optimizer_settings, id=cell_id)
    results_out = results
    print("=== Saving Results ===")
    df = pd.DataFrame(results_out, index=[0])
    df.to_csv(f'output//{cell_id}_spike_fit_opt_CSV.csv')
    min_dict = results_out
    plot_IF(min_dict, model)
    plt.savefig(f"output//{cell_id}_fit_IF.png")
    plot_trace(min_dict, model)
    plt.savefig(f"output//{cell_id}_fit_vm.png")
    
    print("=== Results Saved ===")
    return results

# === Helper functions ===

def load_data_and_model(file, sweep_upper_cut=None):
    ''' loads a NWB file from a file path. Also computes basic parameters such as capactiance   
    takes:  
    file (str): a string pointing to a file to be loaded  
    sweep_upper_cut (int): the sweep at which to stop indexing, sometimes useful to drop low quality sweeps  

    returns:  
    model (b2model): a brian2 model object with the real (in Vitro) data added  
    '''
    
    global dt
    global spiking_sweeps
    global non_spiking_sweeps
    file_path = file
    
    realX, realY, realC,_ = loadNWB(file_path, old=False)
    index_3 = np.argmin(np.abs(realX[0,:]-2.50))
    ind_strt = np.argmin(np.abs((realX[0,:]-0.50)))
    if sweep_upper_cut == None:
        realX, realY, realC = realX[:,ind_strt:index_3], realY[:,ind_strt:index_3], realC[:,ind_strt:index_3]
    else:   
        realX, realY, realC = realX[:sweep_upper_cut,ind_strt:index_3], realY[:sweep_upper_cut,ind_strt:index_3], realC[:sweep_upper_cut,ind_strt:index_3]
    realX = realX - realX[0,0]
    realX, realY, realC = sweepwise_qc(realX, realY, realC)
    sweeplim = np.arange(realX.shape[0])
    dt = compute_dt(realX)
    compute_el = compute_rmp(realY[:2,:], realC[:2,:])
    #baseline the data to the first two sweeps?
    sweepwise_el = np.array([compute_rmp(realY[x,:].reshape(1,-1), realC[x,:].reshape(1,-1)) for x in np.arange(realX.shape[0])])
    sweep_offset = (sweepwise_el - compute_el).reshape(-1,1)
    realY = realY - sweep_offset
    #Compute Spike Times
    spike_time = detect_spike_times(realX, realY, realC, sweeplim, upper=1.15) #finds geh 
    spiking_sweeps = np.nonzero([len(x) for x in spike_time])[0]
    rheobase = spiking_sweeps[0]
    non_spiking_sweeps = np.delete(np.arange(0, realX.shape[0]), spiking_sweeps)[:2]
    thres = compute_threshold(realX, realY, realC, sweeplim)
    #negative current sweeps 
    neg_current = [x<0 for x in realC[:, np.argmin(np.abs(realX-0.5))]]
    neg_current = np.arange(0, realX.shape[0])[neg_current]
    resistance = membrane_resistance_subt(realX[neg_current], realY[neg_current], realC[neg_current])
    taum = exp_decay_factor(realX[0], realY[0], realC[0], plot=True)
    Cm = (mem_cap((resistance*Gohm)/ohm, taum)) * farad
    Cm = Cm/pF
    taum *=1000
    model = brian2_model(param_dict={'EL': compute_el, 'dt':dt, '_run_time':2, 'C': Cm, 'taum': taum})
    model.add_real_data(realX, realY, realC, spike_time, non_spiking_sweeps, spiking_sweeps)
    model.build_params_from_data()
    return model

# === fitting functions ===

def SNPE_OPT(model, optimizer_settings, id='nan', run_ng=True, run_ng_phase=False, run_skopt=False):
    ''' Samples from a SNPE posterior to guess the best possible params for the data. Optionally runs the NEVERGRAD differential
    evolution optimizer restricted to the SNPE-generated top 100 most probable parameter space following sample.
    takes:
    model (a brian2_model object): with the cell data loaded into the objects properties
    id (str): a string containing the cell id. For saving the fit results
    use_post (bool): whether to use the prefit postierior (defaults to true)
    refit_post (bool): whether to refit the prefit postierior (defaults to false)
    run_ng (bool): whether to run a few rounds of the optimizer after sampling from SNPE (defaults to True)
    '''
    from pickle import dump, load

    import torch
    from sbi import utils as utils
    from sbi.inference import SNLE, SNPE
    from sbi.inference.base import infer

    global realC
    global dt
    global non_spiking_sweeps
    global N
    # Generate the X_o (observation) uses a combination of different params
    real_fi, real_isi = compute_FI_curve(model.spike_times, model._run_time) #Compute the real FI curve
    real_fi = np.hstack((real_fi, real_isi))
    real_rmp = compute_rmp(model.realY, model.realC)
    real_min = []
    real_subt = []
    for x in model.subthresholdSweep:
        temp = compute_steady_hyp(model.realY[x, :].reshape(1,-1), model.realC[x, :].reshape(1,-1))
        temp_min = compute_min_stim(model.realY[x, :], model.realX[x,:], strt=0.62, end=1.2)
        real_subt.append(temp)
        real_min.append(temp_min)
    real_subt = np.array(real_subt)        
    
    real_min = np.hstack(real_min)
    N = batch_size
    np_o = np.hstack((real_fi, real_rmp, real_subt, real_min))
    x_o = torch.tensor(np.hstack((real_fi, real_rmp, real_subt, real_min)), dtype=torch.float32)
    
    #adjust the constraints to the found CM or TAUM
    optimizer_settings['constraints'][optimizer_settings['model_choice']]['EL'] = [model.EL*1.01, model.EL*0.99]
    optimizer_settings['constraints'][optimizer_settings['model_choice']]['C'] = [model.C*0.99, model.C*1.01]
    optimizer_settings['constraints'][optimizer_settings['model_choice']]['taum'] = [model.taum*0.99, model.taum*1.01]


    opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']], batch_size, rounds, backend='sbi')
    opt.opt.set_x(x_o)
    #set the default X, seems to speed up sampling
    opt.opt.fit(model, id='test')

    ##TODO maybe take the mode of the highest samples for each column?
    #Take the 
    results_out = {'Cm': params[0], 'taum':params[1], 'EL': params[2], 'a': params[3], 
                    'tauw': params[4], 'VT': params[5], 'VR': params[6],
                    'b': params[7]}
    _labels = list(results_out.keys())
    #Now run a few rounds of optimizer over the data
    top_100 = posterior_samples.numpy()[np.argsort(log_prob)[-100:]] #get the most probable 100 samples
    high = np.apply_along_axis(np.amax,0, top_100) #take the high range of each column (each column is a different parameter)
    low = np.apply_along_axis(np.amin,0, top_100) #take the low range
    var_pairs = np.transpose(np.vstack((low, high))) #stack them into low->pairs for each row
    if run_ng:
        results_out = _opt(model, var_pairs, _labels)  #returns a result containing the param - error matches
    elif run_ng_phase:
        results_out = biphase_opt(model, var_pairs,_labels)
    print("=== Saving Results ===")
    df = pd.DataFrame(results_out, index=[0])
    df.to_csv(f'output//{id}_spike_fit_opt_CSV.csv')
    min_dict = results_out
    plot_IF(min_dict, model)
    plt.savefig(f"output//{id}_fit_IF.png")
    plot_trace(min_dict, model)
    plt.savefig(f"output//{id}_fit_vm.png")
    
    print("=== Results Saved ===")
    return df

def _opt(model, optimizer_settings, optimizer='ng', id='nan'):
        '''
        Runs a nevergrad optimizer (genetic algorithm) over the model and params (https://github.com/facebookresearch/nevergrad). 
        Takes:
        model (a brian2_model object): with the cell data loaded into the objects properties  
        var_pairs: The parameters to be optimized in a [low, high] format. Should be a list of lists or 2d numpy array   
        _labels: The labels for the parameter pairs.   
        id (str): a string containing the cell id. For saving the fit results  
        returns:
        results_out: returned dict of best fit parameters
        '''
        import nevergrad as ng


        model.set_params({'N': batch_size})
        budget = int(rounds * batch_size)

        #adjust the constraints to the found CM or TAUM
        optimizer_settings['constraints'][optimizer_settings['model_choice']]['EL'] = [model.EL*1.01, model.EL*0.99]
        optimizer_settings['constraints'][optimizer_settings['model_choice']]['C'] = [model.C*0.99, model.C*1.01]
        optimizer_settings['constraints'][optimizer_settings['model_choice']]['taum'] = [model.taum*0.99, model.taum*1.01]


        opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']], batch_size, rounds, backend=optimizer, nevergrad_opt=ng.optimizers.ParaPortfolio)#
        error_calc = weightedErrorMetric(weights=[1000, 1])
        min_ar = []
        print(f"== Starting Optimizer with {rounds} rounds ===")
        for i in np.arange(rounds):
            print(f"[CELL {id}] - iter {i} start")
            model.set_params({'N': batch_size, 'refractory':0})
            t_start = time.time()
            param_list = opt.ask()
            param_dict = param_list
            print(f"sim {(time.time()-t_start)/60} min start")
            _, error_t, error_fi, error_isi, error_s = model.opt_full_mse(param_dict)
            test = error_calc.transform(np.vstack([error_fi, error_t]).T)
            error_fi = np.nan_to_num(error_fi, nan=999999) * 500
            error_t  = np.nan_to_num(error_t , nan=999999, posinf=99999, neginf=99999)
            y = error_t + error_fi + error_s
            y = np.nan_to_num(y, nan=999999)
            #y = stats.gmean(np.vstack((error_fi, error_t)), axis=0)
            print(f"sim {(time.time()-t_start)/60} min end")
            opt.tell(param_list, y) ##Tells the optimizer the param - error pairs so it can learn
            t_end = time.time()
            min_ar.append(np.sort(y)[:5])
            res = opt.get_result()
            #try:
            plot_trace(res, model)
                
            plt.savefig(f"output//{id}_{i}_fit_vm.png")
            plot_IF(res, model)
                
            plt.savefig(f"output//{id}_{i}_fit_IF.png")
                #os.remove(f"output//{id}_{i-1}_fit_vm.png")
                #os.remove(f"output//{id}_{i-1}_fit_IF.png")
            #except:
               # pass
            if len(min_ar) > 5:
                 if _check_min_loss_gradient(min_ar, num_no_improvement=25, threshold=1e-5) == False:
                     break
            print(f"[CELL {id}] - iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)} with a min trace error {error_t[np.argmin(y)]} and FI error of {error_fi[np.argmin(y)]} and spike error of {error_s[np.argmin(y)]}") #    
        results = opt.get_result() #returns a result containing the param - error matches
        results_out = results
        return results_out

def biphase_opt(model, var_pairs, _labels, optimizer='ng', id='nan'):
        ''' Run a biphasic'''
        
        
        import nevergrad as ng



        model.set_params({'N': batch_size})
        var_pairs_subt = var_pairs[:5,:]
        budget = int(rounds * batch_size)
        

        opt = snmOptimizer(var_pairs_subt, _labels[:5], batch_size, rounds, backend=optimizer, nevergrad_opt= ng.optimizers.ParaPortfolio)
        min_ar = []
        print(f"== Starting Optimizer with {rounds} rounds ===")
        for i in np.arange(rounds):
            print(f"iter {i} start")
            model.set_params({'N': batch_size, 'refractory':0, 'VT': 0})
            t_start = time.time()
            param_list = opt.ask()
            print(f"sim {(time.time()-t_start)/60} min start")
            error_t = model.opt_trace(param_list)
            y = np.nan_to_num(error_t, nan=9999) 
            print(f"sim {(time.time()-t_start)/60} min end")
            opt.tell(param_list, y) ##Tells the optimizer the param - error pairs so it can learn
            t_end = time.time()
            min_ar.append(np.sort(y)[:5])
            if len(min_ar) > 25:
                 if _check_min_loss_gradient(min_ar, num_no_improvement=25) == False:
                     break
            res = opt.get_result()
            try:
                plot_trace(res, model)
                
                plt.savefig(f"output//{id}_{i}_fit_vm.png")
                os.remove(f"output//{id}_{i-1}_fit_vm.png")
            except:
                pass
            print(f"iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)} with a min trace error {error_t[np.argmin(y)]} ") #    
        results = opt.get_result()  #returns a result containing the param - error matches
        #Now intialize for F-I fitting
        #fix to the best fit values
        model.set_params(results)
        model.set_params({'N': batch_size})
        
        budget = int(rounds * batch_size)
        opt = snmOptimizer(var_pairs[5:], _labels[5:], batch_size, rounds, backend=optimizer, nevergrad_opt=ng.optimizers.ParaPortfolio)
        min_ar = []
        for i in np.arange(rounds):
            print(f"iter {i} start")
            model.set_params({'N': batch_size, 'refractory':0})
            t_start = time.time()
            param_list = opt.ask()
            print(f"sim {(time.time()-t_start)/60} min start")
            error_fi = model.opt_FI(param_list)
            y = error_fi
            print(f"sim {(time.time()-t_start)/60} min end")
            opt.tell(param_list, y) ##Tells the optimizer the param - error pairs so it can learn
            t_end = time.time()
            min_ar.append(np.sort(y)[:5])
            res = opt.get_result()
            try:
                plot_trace(res, model)
                
                plt.savefig(f"output//{id}_{i}_fit_vm.png")
                plot_IF(res, model)
                
                plt.savefig(f"output//{id}_{i}_fit_IF.png")
                os.remove(f"output//{id}_{i-1}_fit_vm.png")
                os.remove(f"output//{id}_{i-1}_fit_IF.png")
            except:
                pass
            print(f"iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)} with a min FI error {error_fi[np.argmin(y)]} ")
            if len(min_ar) > 25:
                 if _check_min_loss_gradient(min_ar, num_no_improvement=25) == False:
                     break

    
        results.update(opt.get_result())

        results_out = results
        print("=== Saving Results ===")
        df = pd.DataFrame(results_out, index=[0])
        df.to_csv(f'output//{id}_spike_fit_opt_CSV.csv')
        min_dict = results_out
        plot_IF(min_dict, model)
        plt.savefig(f"output//{id}_fit_IF.png")
        plot_trace(min_dict, model)
        plt.savefig(f"output//{id}_fit_vm.png")
        
        print("=== Results Saved ===")
        return results_out






if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Single Fit a NWB')

    _dir = os.path.dirname(__file__)
    parser.add_argument('--inputFile', type=str, 
                        help='The NWB file to be fit', required=True)
    parser.add_argument('--outputFolder', type=str,
                        help='the output folder for the generated data', default= _dir +'//output//')
    parser.add_argument('--optimizer', type=str, default='ng',
                        help='the optimizer to use', required=False)
    args = parser.parse_args()
    run_optimizer(args.inputFile, optimizer=args.optimizer)

    


# Functions below here I need to update or document or delete so just ignore for now

def adif_model(cm_i, taum_i, El_i, a_i, tau_w_i,  Vt_i, VR_i, b_i, realC_i=None, record_v=False) -> [StateMonitor, SpikeMonitor]:
    '''
    Simple adif Model function that takes param inputs and outputs the voltage and spike times
    ---
    Takes:
    For below the inputs are in array shape (num_units,), where num_units is the number of realizations of the simulation
    cm_i (numpy array): Cell Capacitance (cm) in picofarad 
    taum_i (numpy array): Cell time Constant in ms
    El_i (numpy array): Leak potential in mV
    tau_w_i (numpy array): Time constant for adaptation in ms
    Vt_i (numpy array): Voltage threshold for registering a spike in mV
    VR_i (numpy array): Reset potential for Vm post-spike in mV
    

    realC_i (numpy array): 1D numpy array representing the input current in picoamp
    record_v (bool): Whether to record the voltage for all cells (true), or not (false)

    Returns:
    voltage (brian2 voltage monitor) with shape (num_units, time_steps)
    spike times (brian2 spike monitor)

    '''
    start_scope()
    eqs='''
    dv/dt = ( gL*(EL-v) + I - w ) * (1./Cm) : volt (unless refractory)
    dw/dt = ( a*(v - EL) - w ) / tauw : amp
    tauw : second
    a : siemens
    b : amp
    Cm : farad
    taum : second
    gL : siemens
    EL : volt
    VT : volt
    VR : volt
    I = in_current(t) : amp
    '''
    global dt
    
    in_current = TimedArray(values = realC_i * pamp, dt=dt * ms)
    G1 = NeuronGroup(N, eqs, threshold='v > VT', reset='v = VR; w+=b', refractory=0 * ms, method='euler')
    #init:
    G1.tauw = tau_w_i *ms; 
    G1.b = b_i * pA; 
    G1.a = a_i * nS; 
    G1.VT = Vt_i * mV;
    G1.taum = taum_i * ms;
    G1.VR = VR_i *mV
    #parameters
    G1.Cm = cm_i * pF
    G1.gL = ((cm_i *pF)/ (taum_i * ms))
    G1.EL = El_i *mV
    G1.v = El_i *mV
    # record variables
    if record_v == True:
        #Only record voltage if explicity asked to save memory
        Mon_v = StateMonitor( G1, "v", record=True, dt= dt * ms)
    else:
        Mon_v = None
    Mon_spike = SpikeMonitor( G1 )
    run(2.5 * second)
    return Mon_v, Mon_spike

def generate_fi_curve(param_set, save=True) -> torch.tensor:
    ''' Function to handle interfacing between SNPE and brian2, running each sweep and computing the FI curve.
    Takes:
    param_set (torch.tensor) of shape (num_units, params): A tensor containg the randomly sampled params, each row represents a single cell param set.

    Returns:
    vars (torch.tensor) of shape (num_units, measured vars): A tensor containing the FI curve, ISI-mode-Curve (most common ISI per Sweep), and subthreshold params
    '''
    global non_spiking_sweeps
    param_list = param_set.numpy() #Input is pytorch tensor converting it to a numpy array for brian2 purposes
    if param_list.shape[0] > 12:
        cm_i, taum_i, El_i, a_i, tau_w_i, Vt_i, VR_i, b_i = np.hsplit(np.vstack(param_list), 8) #Splits the variables out of the array
        ## Then flatten the variable arrays
        cm_i, taum_i, El_i, a_i, tau_w_i, Vt_i, VR_i, b_i = np.ravel(cm_i), np.ravel(taum_i), np.ravel(El_i), np.ravel(a_i), np.ravel(tau_w_i), np.ravel(Vt_i), np.ravel(VR_i), np.ravel(b_i)
    else:
        cm_i, taum_i, El_i, a_i, tau_w_i, Vt_i, VR_i, b_i = param_list
    spikes_full = [[] for x in np.arange(N)]
    isi_full = [[] for x in np.arange(N)]
    ##Generate an empty list of length N_UNITS which allows us to dump the subthreshold params into
    subthres_features = [[] for x in np.arange(N)]
    rmp_full = [[] for x in np.arange(N)]
    stim_min = [[] for x in np.arange(N)]
    for i, sweep in enumerate(realC): #For each sweep
        print(f"Simulating sweep {i}")
        voltage, spikes = adif_model(cm_i, taum_i, El_i, a_i, tau_w_i, Vt_i, VR_i, b_i, realC_i=sweep, record_v=True) #Run the adex model with the passed in params
        temp_spike_array = spikes.spike_trains() # Grab the spikes oriented by neuron in the network
        print("Simulation Finished")
        print("Computing features")
        for p in np.arange(N): #For each neuron
                pspikes = temp_spike_array[p] #Grab that neurons spike times
                if len(pspikes) > 0: #If there are any spikes
                    spikes_full[p].append(len(pspikes)) #Count the number of spikes and add it to the full array
                    spike_s = pspikes/ms #get the spike time in ms
                    if len(spike_s) > 1: #If there is more than one spike
                        isi_full[p].append(np.nanmean(np.diff(spike_s))) #compute the mode ISI
                    else:
                        isi_full[p].append(0) #otherwise the mode ISI is set to zero
                else:
                    spikes_full[p].append(0) #If no spikes then send both to zero
                    isi_full[p].append(0)

                
                ##Compute Subthresfeatures
                temp_rmp = compute_rmp(voltage[p].v/mV.reshape(1,-1), sweep.reshape(1,-1)) #compute the beginning Resting membrane
                if i in non_spiking_sweeps:
                    temp_deflection = compute_steady_hyp(voltage[p].v/mV.reshape(1,-1), sweep.reshape(1,-1)) #compute the end
                    subthres_features[p].append(np.hstack((temp_rmp, temp_deflection)))

                    #compute Sweepwisemin
                    temp_min = compute_min_stim(voltage[p].v/mV, voltage[0].t/second, strt=0.62, end=1.0)
                    stim_min[p].append(temp_min)
        print("Computation finished")
    neuron_avgs = np.vstack([np.nanmean(np.vstack(x), axis=0) for x in subthres_features])
    #of the SubT features
    spikes_return = np.array(spikes_full) #Stack all the arrays together
    isi_return = np.array(isi_full) #Stack all the arrays together
    min_return = np.array(stim_min)
    sub_return = np.array(subthres_features)
    return_full = np.hstack((spikes_return / 2, isi_return, neuron_avgs, min_return))
    #Load and save array
    
    return torch.tensor(return_full, dtype=default_dtype)

def _check_min_loss_gradient(min_ar, num_no_improvement=10, threshold=1e-5):
    pass_check = True
    min_ar = np.array(min_ar)
    min_ar = np.nanmean(min_ar, axis=1)
    if min_ar.shape[0] <= num_no_improvement:
        pass
    else:
        x = np.arange(0, num_no_improvement)
        slope, _, _, p, _ = stats.linregress(x, min_ar[-num_no_improvement:])
        if (slope < threshold and slope > -threshold) and p < 0.01:
            pass_check = False
        print(slope)
    
    return pass_check

