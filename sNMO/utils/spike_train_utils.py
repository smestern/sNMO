import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from brian2 import *
try:
    import networkx as nx
except:
    print("networkx not installed, some functions will not work")

try: 
    from elephant.statistics import cv2
    from neo import SpikeTrain
except:
    print("elephant not installed, some functions will not work")
    SpikeTrain = np.ScalarType

#% MISC utils %#
#Tools for loading or transforming spike trains

def cast_backend_spk(spike_trains, check_each=False):
    """Checks the backend of the spike trains, and transforms them into a list of numpy arrays, should be used for all functions.
    removes brian2 units and brian2 backends
    takes:
        spike_trains: the spike trains to check. Can be a list of spike trains, a list of lists of spike trains, a dict of spike trains, a brian2 spike monitor, or a neo spike train
        check_each: whether to check each spike train individually, or assume they are all the same
    returns:
        spike_trains: a list of spike trains"""
    #should transform individual spike trains into a list of spike trains
    if isinstance(spike_trains, list) or isinstance(spike_trains, np.ndarray):
        #check the shape/len
        if len(spike_trains) < 1: #if the list is empty, return it but check for units
            if isinstance(spike_trains, Quantity):
                spike_trains = spike_trains / second
            else:
                spike_trains = np.array(spike_trains)
        elif isinstance(spike_trains[0], list) or isinstance(spike_trains[0], np.ndarray):
            #check if the spike trains are in brian2 format e.g. have units
            #this may be the bottom of the recursion, so check the shape/len
            if isinstance(spike_trains[0], Quantity):
                spike_trains = [x / second for x in spike_trains] # probably a single spike train, so just remove the units
                #make into an array
                spike_trains = np.array(spike_trains)
            elif isscalar(spike_trains[0]):
                spike_trains = np.array(spike_trains) #do nothing if it is a scalar, likely a list of spike times
            elif len(spike_trains[0]) <= 1:
                spike_trains = [np.array(x / second) for x in spike_trains]
            else:
                spike_trains = [cast_backend_spk(x) for x in spike_trains]   #recursively call the function
    elif isinstance(spike_trains, SpikeMonitor): #check if the spike trains are in brian2 format
        spike_trains = spike_trains.spike_trains()
        #this is now a dict of spike trains, we need to transform it into a list of spike trains
        spike_trains = [spike_trains[x]/second for x in spike_trains.keys()] #also remove the b2 untis
    #check if the spike trains are in neo format
    elif isinstance(spike_trains, SpikeTrain):
        spike_trains = [spike_trains]
    elif isinstance(spike_trains, dict):
        #check if the spike trains are in brian2 format e.g. have units
        if isinstance(spike_trains[list(spike_trains.keys())[0]][0], Quantity):
            spike_trains = [ np.array(v / second)for k, v in spike_trains.items()]
        else:
            spike_trains = [ np.array(v) for k, v in spike_trains.items()]
    else:
        raise Exception("Spike trains not in a recognized format")
    return spike_trains
    
def build_n_train(isi, n=2, padding=np.nan, truncate=True):
    """Builds a n-dimensional isi array from a 1D isi array. For use in the swasserstein distance functions, and the joint-ISI plots
    takes:
        isi: the isi array in ms
        n: the number of dimensions to build
        padding: the value to pad the array with
    returns:
        isi_corr: the n-dimensional isi array"""
    isi_corr = isi

    if np.isscalar(padding):
        pass
    elif callable(padding):
        padding = padding(isi_corr)
    else:
        raise Exception("Padding must be a number or a function")

    for p in np.arange(1,n):
         temp = np.hstack((isi, np.full(p, np.nan)))[p:]
         isi_corr = np.vstack((isi_corr, temp))
    isi_corr_out = np.nan_to_num(isi_corr, nan=padding)

    if truncate:
        isi_corr_out = isi_corr_out.T[:-p, :]
    return isi_corr_out

def build_isi_from_spike_train(spike_trains, low_cut=0., high_cut=None, indiv=True):
    """ Builds an ISI array from a spike train. This is used for the joint-ISI plots.
    takes:
        spike_trains: the spike trains to build the ISI array from
        low_cut: the low cut off for the ISI array
        high_cut: the high cut off for the ISI array
        indiv: whether to return the ISI array as a list of arrays, or a single array
    returns:
        ISI_: the ISI array"""
    spike_trains = cast_backend_spk(spike_trains)
    ISI_ = []
    N = len(spike_trains)
    if high_cut is None:
        for u in np.arange(N).astype(np.int32):
            spikes = spike_trains[u]
            spikes_filtered = spikes[spikes>low_cut]
            ISI_.append( np.diff(spikes_filtered)*1000)
    else:
        for u in np.arange(N).astype(np.int32):
            spikes = spike_trains[u]
            spikes_filtered = spikes[np.logical_and(spikes>low_cut, spikes<high_cut)]
            ISI_.append(np.diff(spikes_filtered)*1000)
    
    if indiv==False:
        ISI_ = np.hstack(ISI_)
        if ISI_.shape[0] < 1:
            ISI_ = np.array([3000,3000])
    return ISI_

isi_from_spike_train = build_isi_from_spike_train #alias to old function name

def unweave_spikes(times, ids):
    """Unweave spikes in a given time array with corresponding ids"""
    ids_ = np.unique(ids)
    times_ = []
    for x in ids_:
        times_.append(times[ids==x])
    return times_, ids_

def weave_spikes(times, ids=None):
    """Weave spikes in a given time array with corresponding ids"""
    times_ = np.hstack(times)
    if ids is None:
        ids_ = np.hstack([np.full(len(x), i) for i, x in enumerate(times)])
    else:
        ids_ = np.hstack(ids)
    return times_, ids_


## ICHIYAMA ET AL. 2022 summary functions

def inter_burst_hist(isi,bins=np.logspace(0,5), low_cut=None):
    """ Computes the histogram of the interburst intervals, and returns the raw interburst intervals
    takes:
        isi: the isi array in ms
        low_cut: the low cut off for the interburst intervals
        bins: the bins to use for the histogram
    returns:
        inter_burst: the histogram of the interburst intervals
        burst_times: the raw interburst intervals
    """

    labeled_isi, bursts, interburst = filter_bursts(isi, label_sil=True) #get the bursts
    burst_times = np.cumsum(isi)[np.where(labeled_isi == 3)[0]] #get the burst times
    if low_cut is not None: #legacy code, not sure if this is needed, ideally should be removed
        burst_isis = isi[np.where(labeled_isi == 3)[0]]
        burst_times = burst_times[burst_isis>low_cut]

    burst_times = burst_times[1:] - burst_times[:-1] #get the interburst intervals
    inter_burst = np.histogram(burst_times, bins)[0] #compute the histogram
    inter_burst = inter_burst / inter_burst.sum() #normalize the histogram
    return inter_burst, burst_times


def inter_burst_mean(isi, low_cut=10):
    inter_burst, pre_burst = inter_burst_hist(isi, low_cut=low_cut)
    mean = np.nanmean(pre_burst)
    median = np.nanmedian(pre_burst)
    return mean, median

def preceeding_sil_per_event_len(isi, event_len_bins=np.arange(0, 6), silence=3):
    """
    Computes the preceeding silence for each event length. This is the time between bursts based on how many spikes are in the burst.
    takes:
        isi: the isi array in ms
        event_len_bins: the bins to use for the event length
    returns:
        event_len_burst_times: the preceeding silence for each event length (list of arrays)
        event_len_uni: the unique event lengths
    """
    labeled_isi, bursts, interburst = filter_bursts(isi, sil=silence, label_sil=True)
    #measure the time between bursts, this is the time between 3's in the labeled isi
    burst_times = isi[np.where(labeled_isi == 3)[0]]/1000 #this is the silences before the burst
    assert len(burst_times) == len(bursts) #make sure we have the same number of bursts as we do silences
    #binarize the event length
    event_length = [len(burst)-1 for burst in bursts] #since our bins are 0-5, we need to subtract 1 from the burst length
    event_length = np.digitize(event_length, event_len_bins, right=True)
    #bin the burst times based on the event length
    event_len_burst_times = []
    for x in event_len_bins:
        if x != event_len_bins[-1]: #open interval final bin
            event_len_burst_times.append(burst_times[event_length==x])
        else:
            event_len_burst_times.append(burst_times[event_length>=x])
    return event_len_burst_times, event_len_bins

def prob_burst_per_isi(isi, bins=np.logspace(0,5), min_isi=3):
    """
    Computes the probability of a burst occuring after each isi. This is the probability of a burst occuring after each isi.
    takes:
        isi: the isi array in ms
        bins: the bins to use for the isi
        min_isi: the minimum isi to use for the probability
    returns:
        results: the probability of a burst occuring after each isi
    """
    results = np.zeros(len(bins))
    labeled_isi, bursts, interburst = filter_bursts(isi, sil=min_isi, label_sil=False) # here we want to be unfiltered, so we set the threshold to min_isi
    #now we need to bin the isi's and count the number of bursts that occur after each isi
    digitize = np.digitize(isi, bins) #here we digitize the isi in
    for bin_idx in np.unique(digitize):
        idxs = np.where(digitize == bin_idx)[0]+1
        idxs = idxs[idxs < len(labeled_isi)]
        num_bursts = np.sum(labeled_isi[idxs] == 1)
        results[bin_idx-1] = num_bursts / len(idxs)
    return results

def tonic_hist(isi):
    train_ = build_n_train(isi)
    less_ = np.logical_and(train_[:,0]>=6,train_[:,1]>=6)
    bins = np.logspace(0,4)
    pre_burst = train_[less_, 0]
    intra_burst = np.histogram(pre_burst, bins)[0]
    intra_burst = intra_burst / intra_burst.sum()
    return intra_burst, pre_burst

def binary_spike_train(TIME, bin_size=5, end=None, binary=True):
    '''Converts Time array into binary spike train
    all units should be in ms'''
    if end is None:
        end = TIME[-1] + bin_size
    bins = np.arange(0, end, bin_size)
    hist,_ = np.histogram(TIME, bins)
    if binary:
        hist[hist>1] = 1
    return hist

def binned_fr(spike_times, end_time, binsize=0.5, bins=None):
    """ Returns the binned firing rate of a single neuron. Spike times can either be a list of spike times or a list of spike trains.
    takes:
        spike_times: list of of spike times (in seconds)
        end_time: the end time of the simulation (in seconds)
        binsize: the size of the bins (in seconds)
    returns:
        binned_isi_hz: the binned firing rate of the neuron
        bins: the bins used for the binned firing rate
    """
    if bins is None:
        bins = np.arange(0, end_time, binsize)
    if bins is not None:
        binsize = bins[1] - bins[0]
    bins_right = np.arange(binsize,end_time + binsize, binsize)
    #cast the backend
    spike_times = cast_backend_spk(spike_times)

    #inner loop for the binned firing rate
    def _inner_bin_loop(inner_spike):
        inner_binned_isi_hz = []  
        for x, x_r in zip(bins, bins_right):
            filtered_isi = inner_spike[np.logical_and(inner_spike>=x, inner_spike<x_r)]
            inner_binned_isi_hz.append((len(filtered_isi)/(binsize)))
        return np.hstack(inner_binned_isi_hz)

    #try to figure out if spike_times is a list of spike trains or a single neuron list of spike times
    if len(spike_times) > 1:
        #check the inner shape
        if np.isscalar(spike_times[0]):
            #this is a list of spike times from a single neuron (probably)
            #logging.debug("binned_fr is being called on a list of spike times")
            binned_isi_hz = _inner_bin_loop(spike_times)
        elif isinstance(spike_times[0], np.ndarray): #probably a list of spike trains
            #print("binned_fr is being called on a list of spike trains, did you mean to call binned_fr_pop?")
            binned_isi_hz = [_inner_bin_loop(x) for x in spike_times]
            #vstack the binned_isi_hz
            binned_isi_hz = np.vstack(binned_isi_hz)
    elif len(spike_times) < 1:
        #this is an empty neuron
        binned_isi_hz = np.zeros(len(bins))
    else:
        #this is a single neuron
        binned_isi_hz = _inner_bin_loop(spike_times[0])
    return binned_isi_hz, bins


def binned_fr_pop(spike_times, end_time, popsize=None, binsize=0.5):
    """ Returns the binned firing rate of a population of neurons. Spike times can either be a list of spike times or a list of spike trains.
    takes:
        spike_times: list of spike times or list of spike trains (in seconds)
        end_time: the end time of the simulation (in seconds)
        popsize: the number of neurons in the population (default 500)
        binsize: the size of the bins (in seconds)
    returns:
        binned_isi_hz: the binned firing rate of the population
        bins: the bins used for the binned firing rate
    """
    spike_times = cast_backend_spk(spike_times)
    if popsize is None:
        popsize = len(spike_times)
    binned_isi_hz = []
    bins = np.arange(0, end_time, binsize)
    bins_right = np.arange(binsize,end_time + binsize, binsize)
    for x, x_r in zip(bins, bins_right):
        temp_isi_array = []
        for u in np.arange(popsize):
            temp_isi = spike_times[u]
            filtered_isi = temp_isi[np.logical_and(temp_isi>=x, temp_isi<x_r)]
            temp_isi_array.append((len(filtered_isi)/(binsize)))
        binned_isi_hz.append(np.nanmean(temp_isi_array))
    return np.hstack(binned_isi_hz), bins

def equal_ar_size_from_list(isi_list, padding=np.nan):
    """ Takes an array of arrays, and pads them to be the same size. This is used for the joint-ISI plots.
    takes:
        isi_list: the isi array in ms
    returns:
        new_list: the isi array with the same size arrays
    """
    if np.isscalar(padding):
        call_pad = False
    elif callable(padding):
        call_pad = True
    lsize = len(max(isi_list, key=len))
    new_list = []
    for x in isi_list:
        temp_pad = padding(x) if call_pad else padding
        new_list.append(np.hstack((x, np.full((lsize-len(x)), [temp_pad]))))
    return np.vstack(new_list)

def save_spike_train(isi, n=500, rand_num=30, filename='spike_trains.csv'):
    """ Saves a random sample of spike trains from the isi array.
    takes:
        isi: the isi array in ms
        n: the number of spike trains to sample
        rand_num: the number of spikes to sample from each spike train
    returns:
        isi_out: the sampled spike trains
    """

    units_index = np.random.randint(0,n, rand_num)
    isi_trains = []
    for x in units_index:
        train = isi[x] / second
        isi_trains.append(train)
    isi_out = equal_ar_size_from_list(isi_trains)
    np.savetxt(f"{filename}", isi_out.T, fmt='%.18f', delimiter=',')

def save_spike_train_col(isi, filename='spike_trains_col.csv'):
    train = isi.i
    time = isi.t
    isi_out = np.vstack((train, time))
    np.savetxt(f"{filename}", isi_out.T, fmt='%.18f', delimiter=',')


def filter_bursts(isi, sil=25, burst=6, intraburst=25, label_sil=False, binary=False, only_initial=False):
    """ Filters the bursts, and 'tonic' spiking of the the given isi array. labelling the spike train where appropriate. Default values are
    from the Ichiyama et al. 2021 paper. Advanced method allows for an exponeniatal growth of the intraburst time, but this is not
    implemented yet.
    takes:
        isi: the isi array in ms
        sil: the silence time (ms)
        burst: the burst time (ms)
        intraburst: the intraburst time (ms)
        label_sil: whether to label the silence as part of the burst or not. the silence will not be included in the bursts list either way
        binary: whether to return a binary labeled isi array or not
    returns:
        labeled_isi: the isi array with 0 for non-burst, 1 for burst, 2 for intraburst
        bursts: a list of bursts
        non_burst: a list of non-burst spikes
    """
    n_train = build_n_train(isi, 2)
    silences_ind = np.where(np.logical_and(n_train[:,0]>=sil, n_train[:,1]<=burst))[0] #intial burst should have a silence of sil ms, and a burst of burst ms
    labeled_isi = np.zeros(isi.shape[0]) #0 is no burst, 2 is intraburst, 1 is burst
    bursts = [] #now we need to find the 'end' of the burst and label the intraburst
    for x in silences_ind: #for each silence
        temp_isi = isi[int(x)+1:] #get the isi after the silence
        end_x = x+1 #start the start of the burst
        for i in temp_isi: #for each isi after the silence
            if i >= intraburst: #if the isi is greater than the intraburst time
                break
            else:
                end_x += 1 #otherwise increment the end of the burst
        labeled_isi[int(x)+1:end_x]=2 #label the intraburst
        bursts.append(isi[int(x):end_x]) #append the burst to the list
    labeled_isi[silences_ind+1] = 1 #label the burst
    #if we want to label the silence as part of the burst
    if label_sil:
        labeled_isi[silences_ind] = 3 #label the silence, as part of the burst
    non_burst = isi[labeled_isi==0] #get the non burst
    #if 
    if binary and not only_initial:
        labeled_isi = np.clip(labeled_isi, 0, 1)
    elif binary and only_initial:
        labeled_isi[labeled_isi==2] = 0
        labeled_isi[labeled_isi==3] = 0
    return labeled_isi, bursts, non_burst

def filter_tonic(isi, sil=25, burst=6, intraburst=50):
    """ Filters out bursts and intrabursts, leaving only 'tonic' spikes. This is the opposite of filter_bursts, and 
    is a bit of a hack. The criteria is that the isi before and after the spike is greater than the intraburst time.
    takes:
        isi: the isi array in ms
        sil: the silence time (ms)
        burst: the burst time (ms)
        intraburst: the intraburst time (ms)
    returns:
        labeled_isi: the isi array with 0 for non-burst, 1 for burst, 2 for intraburst
        bursts: a list of bursts
        non_burst: a list of non-burst spikes

    """

    #first we actually need to find the bursts
    labeled_bursts, _, _ = filter_bursts(isi, label_sil=False, binary=True)


    n_train = build_n_train(isi, 2)
    silences_ind = np.where(np.logical_and(n_train[:,0]>=sil, n_train[:,1]<=burst))[0] #intial burst should have a silence of sil ms, and a burst of burst ms
    probable_tonic = np.where(np.logical_and(n_train[:,0]>=intraburst, n_train[:,1]>=intraburst))[0]
    
    labeled_isi = np.zeros(isi.shape[0]) #0 is no burst, 2 is intraburst, 1 is burst, 3 is silence
    labeled_isi[probable_tonic] = 1


    labeled_isi[silences_ind] = 0 # silences that pre
    #also need to remove the bursts
    labeled_isi[labeled_bursts==1] = 0
    tonic_spikes = isi[labeled_isi==1]
    non_tonic = isi[labeled_isi==0]
    return labeled_isi, tonic_spikes, non_tonic

def compute_states(isi, sil=25, burst=6, intraburst=25, tonic_state_thres = 2, burst_state_thres = 0.05,
                bins=None, binwidth=3000, rolling_mean=True, binary=True, rm_kwargs={'window_n': 3, 'padding': np.nanmean}):
    """
    Computes the states of the ISI array, using the filter_bursts and filter_tonic functions. Returns the "states" in the context of the Ichiyama
    et al 2021. and Mestern et al. 2023 (in prep) papers. The states are: tonic, burst, intraburst, and unclassified. The states are computed using the
    filter_bursts and filter_tonic functions. The states are then binned using the bin_signal function. The bins are by default 3000 ms, but can be
    changed. The binning is done using a rolling mean, but this can be changed.
    takes:
        isi: the isi array in ms (1D array of [n_spikes])
        sil: the silence time (ms)
        burst: the burst time (ms)
        intraburst: the intraburst time (ms)
        tonic_state_thres: the threshold for the tonic state
        burst_state_thres: the threshold for the burst state
        bins: the bins to bin the signal into (1D array of [n_bins]) in ms
        binwidth: the width of the bins in (ms)
        rolling_mean: whether to use a rolling mean or not
        binary: whether to return a binary labeled isi array or not
        rm_kwargs: kwargs for the rolling mean function
    returns:
        binned_states: the binned labels (1D array of [n_bins]), 0 is unclassified, 1 is tonic, 2 is intraburst, 3 is burst
        binned_burst: the binned burst signal (1D array of [n_bins])
        binned_tonic: the binned tonic signal (1D array of [n_bins])

    """
    if bins is None:
        bins = np.arange(0, np.max(np.cumsum(isi)), binwidth)
    elif bins is not None:
        bins = bins
        binwidth  = bins[1] - bins[0]

    #filter the bursts and tonic spikes
    labelb, bursts, non_bursts = filter_bursts(isi, burst=burst, sil=sil, intraburst=intraburst, binary=True, only_initial=True)
    labelt, tonic, non_tonic = filter_tonic(isi, sil=sil, intraburst=intraburst,)
    _, binned_burst = bin_signal(np.cumsum(isi), labelb, bins=bins, bin_func=np.sum)
    _, binned_tonic = bin_signal(np.cumsum(isi), labelt, bins=bins, bin_func=np.sum)

    #transform into a rate
    binned_burst = binned_burst / (binwidth /1000) #convert to rate
    binned_tonic = binned_tonic / (binwidth /1000) #convert to rate

    #compute the rolling means
    if rolling_mean:
        rolling_bins, binned_burst = rolling_window(bins, binned_burst, **rm_kwargs)
        binned_burst = np.nanmean(binned_burst, axis=1)
        rolling_bins, binned_tonic = rolling_window(bins, binned_tonic, **rm_kwargs)
        binned_tonic = np.nanmean(binned_tonic, axis=1)
    
    #compute the mean of the states
    #classify the state as either burst or tonic
    state = np.zeros(binned_burst.shape)
    tonic_states = np.where(np.logical_and(binned_burst <= burst_state_thres, binned_tonic >= tonic_state_thres))[0]
    burst_states = np.where(np.logical_and(binned_burst >= burst_state_thres, binned_tonic <= tonic_state_thres))[0]
    #transition states are not used in the paper, but are included here for completeness
    transition_states = np.where(np.logical_and(binned_burst >= burst_state_thres, binned_tonic >= tonic_state_thres))[0]
    #silence_states = np.where(np.logical_and(binned_burst <= burst_state_thres, binned_tonic < tonic_state_thres))[0] commented out as this is not used in the paper
    #also this is basically the same as 0
    state[transition_states] = 2
    state[tonic_states] = 1
    state[burst_states] = 3

    state = np.array(state, dtype=int)

    return state, binned_burst, binned_tonic
    

def spikes_per_burst(isi):

    labels, burst, non_burst = filter_bursts(isi)
    lens = []
    for x in burst:
        lens.append(len(x)+1)
    bins = np.arange(2,max(lens))
    hist = np.histogram(lens, bins)
    return lens, hist, bins


def rolling_window(X, Y, window_n, padding=np.nanmean,):
    """
    Rolling window for 1D data.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    isi_corr = Y
    for p in np.arange(1, window_n):
         temp = np.hstack((Y, np.full(p, np.nan)))[p:]
         isi_corr = np.vstack((isi_corr, temp))
    
    #if the padding is a function, apply it to the data
    if callable(padding):
        padding = padding(isi_corr)
    #if the padding is a number, fill with that number
    Y_ = np.nan_to_num(isi_corr, nan=padding)
    return X[:-1], Y_.T


def bin_signal(X, Y, bins=None, bin_func=np.nanmean, padding=np.nanmean):
    """
    Bin signal.
    """
    if bins is None:
        bins = np.arange(np.min(X), np.max(X), 0.1)
    else:
        bins = bins

    if padding is not None and bin_func is np.nanmean:
        #pad y with padding function if 
        if callable(padding):
            pad_val = padding(Y)
        else:
            pad_val = padding
        Y = np.hstack((np.full(bins.shape, pad_val), Y, np.full(bins.shape, pad_val)))
        X = np.hstack((np.full(bins.shape, bins[0]-(1e-12)), X, np.full(bins.shape, bins[-1])))

    Y_ = np.zeros(len(bins))
    for i in np.arange(len(bins)-1):
        Y_[i] = bin_func(Y[(X > bins[i]) & (X <= bins[i+1])])
    #fill the last bin
    #Y_[-1] = bin_func(Y[X > bins[-1]])
    return bins, np.nan_to_num(Y_)


def statistic_spikes(spikes, stat_fun=np.mean, window=10):
    """ Computes statsicis of the provided spike trains. This is used for the CV2 and ISI CV2 plots. as well as the rolling fr
    takes:
        spikes: the spike trains to compute the statistics from (in seconds)
        stat_fun: the statistic to compute
        window: the window to use for the rolling window
    returns:
        bins: the bins used for the rolling window
        cv2_: the computed statistic
    """
    spikes = cast_backend_spk(spikes)
    cv2_ = []
    for unit in np.arange(len(spikes)):
        isi = np.diff(spikes[unit]/ms)
        isi_rolling_x, isi_rolling_y = rolling_window(spikes[unit], isi, window, 0)
        cv2_temp = np.apply_along_axis(stat_fun, 1, isi_rolling_y)
        cv2_.append(bin_signal(isi_rolling_x, cv2_temp, bins=np.arange(5, 800.5, 1))[1])
    bins, _ = bin_signal(isi_rolling_x, cv2_temp, bins=np.arange(5, 800.5, 1))
    cv2_ = np.vstack(cv2_)
    return bins, cv2_


def build_networkx_graph(EI_Conn, IE_Conn, N=1000, I_strt_idx=500):
    """Builds a networkx graph from the EI and IE connections
    Takes:
    EI_conn: a list of tuples of the form (E, I)
    IE_conn: a list of tuples of the form (I, E)
    I_st_idx: The reference index for the I neurons
    """
    if I_strt_idx is None:
        I_strt_idx = np.amax(EI_Conn[:,0])

    G = nx.DiGraph()
    G.add_nodes_from(np.arange(N))

    #figure out if the use accidentally passed in the wrong shape
    if EI_Conn.shape[1] == 2:
        pass
    elif EI_Conn.shape[0] == 2 and EI_Conn.shape[1] != 2:
        #transpose the array
        print("EI_conn we think is (2, N) instead of (N, 2), transposing")
        EI_Conn = EI_Conn.T
    else:
        print("EI_conn is not the right shape, we think it should be (N, 2), but it might not be")
    
    if IE_Conn.shape[1] == 2:
        pass
    elif IE_Conn.shape[0] == 2 and IE_Conn.shape[1] != 2:
        #transpose the array
        print("IE_conn we think is (2, N) instead of (N, 2), transposing")
        IE_Conn = IE_Conn.T
    else:
        print("IE_conn is not the right shape, we think it should be (N, 2), but it might not be")
    
    EI_adjusted = np.column_stack((EI_Conn[:, 0],EI_Conn[:, 1] + I_strt_idx))
    IE_adjusted = np.column_stack((IE_Conn[:, 0] + I_strt_idx,IE_Conn[:, 1]))
    full_connection_array = np.vstack((EI_adjusted, IE_adjusted))
    G.add_edges_from(full_connection_array)
    return G


def build_kde(isi_corr):
    isi_corr = np.log10(isi_corr)[:-1]
    unit_kde = stats.gaussian_kde(isi_corr.T)
    return unit_kde

#% Plotting Functions %#
# Functions to plot isi trains in various manners

def plot_patterns(spikes, unit_idx=1, n=500, rand_num=500, figsize=(25,5), fig_num=7, colour=True, random=True):
    """Plots the ISI patterns for a given unit. As presented in Ichiyama et al. 2021
     If n is greater than 1, then it will plot n random units, if random is set to false, then it will plot the first n units
    takes:
        spikes: the spike trains to plot (in seconds)
        unit_idx: the unit to plot
        n: the number of units to plot
        rand_num: the number of spikes to plot
        figsize: the figure size
        fig_num: the figure number
        colour: whether to colour the spikes or not
        random: whether to plot random units or not
    returns:
        fig: the figure object
    """
    spikes = cast_backend_spk(spikes)
    fig = plt.figure(fig_num,  figsize=figsize )
    if random:
        units_index = np.random.randint(0,n, rand_num)
    else:
        if n>1:
            units_index = np.arange(0, unit_idx)
        else:
            units_index = [unit_idx]
    for x in units_index:
        u_isi = spikes[x]
        if len(u_isi) > 0:
            time =  u_isi
            u_isi = (np.diff(u_isi))*1000
            color = ['red', 'blue', 'green', 'purple', 'orange']
            plt.ylim( (1, pow(10,3.5)))
            if colour:
                plt.scatter(time, np.hstack((0,u_isi)), alpha=0.5)
            else:
                plt.scatter(time, np.hstack((0,u_isi)), alpha=0.5, c='grey')
            plt.yscale('log')
            plt.title(f'Unit {x}')
    return fig


def plot_xy(isiarray, lwl_bnd=0, up_bnd=4, fignum=2, color='k', alpha=0.05):
    """Plots the joint-ISI plot for a given unit. As presented in Ichiyama et al. 2021
    takes:
        isiarray: the isi array to plot (in ms)
        lwl_bnd: the lower bound for the plot
        up_bnd: the upper bound for the plot
        fignum: the figure number
        color: the color of the plot
        alpha: the alpha of the plot
    returns:
        fig: the figure object
    """
    plt.figure(fignum, figsize=(10,10))
    if isiarray.shape[0] > 0:
        plt.scatter(isiarray[:,0],isiarray[:,1], marker='o', alpha=alpha, color=color)
    plt.ylim( (pow(10,lwl_bnd), pow(10,up_bnd)) )
    plt.xlim( (pow(10,lwl_bnd), pow(10,up_bnd)) )
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('post isi (ms)')
    plt.xlabel('pre isi (ms)')
    return plt.gcf()

def plot_isi_hist(ISI, fignum=99):
    log_bins = np.logspace(0,4)
    plt.figure(fignum)
    plt.clf()
    plt.hist(ISI, log_bins)
    xscale('log')
    title("Network isi Dist")
    
def plot_intra_burst(isi, fignum=100, low_cut=10):
    intra_burst, pre_burst = inter_burst_hist(isi, low_cut=low_cut)
    log_bins = np.logspace(0,4)
    plt.figure(fignum)
    plt.clf()
    plt.hist(pre_burst, log_bins)
    xscale('log')
    title("Pre Burst Interval")
    return intra_burst

def plot_switching_units(isi, switch, num, burst_thres=0.2, tonic_thres=0.1, n=500, figsize=(25,5), colour=True):
    units_index = np.arange(n)
    good_units = []
    for x in units_index:
        u_isi = isi[x] / second
        if len(u_isi) > 0:
            start_isi = np.diff(u_isi[u_isi<switch]) * 1000
            end_isi = np.diff(u_isi[u_isi>=switch]) * 1000
            if len(start_isi)>0 and len(end_isi)>0:
                bursting = (100>enforce_burst(start_isi, burst_thres))
                no_burst = (100>enforce_no_burst(end_isi, tonic_thres))
                if np.logical_and(bursting, no_burst):
                    good_units.append(x)
    print(len(good_units))
    if len(good_units) >= num:
        units_to_use = np.random.choice(good_units, num, replace=False)
        for x in units_to_use:
                plt.figure(x,  figsize=figsize )
                plt.clf()
                u_isi = isi[x] / second
                time =  u_isi
                u_isi = 1/(np.diff(u_isi))
                color = ['red', 'blue', 'green', 'purple', 'orange']
                plt.ylim( (pow(10,-1), pow(10,3)) )
                plt.scatter(time, np.hstack((0,u_isi)), alpha=0.7, c='k')
                plt.yscale('log')
                plt.title(f'Unit {x}')
    else:
        print("not enough units passing the threshold")
    return len(good_units)
            


#% Distance functions %#
#Various Functions (and internal functions) to measure distances between spike trains
#mostly focused on time-indepedent distances

#MOVED TO SNMO