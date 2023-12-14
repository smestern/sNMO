import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ..utils.spike_train_utils import *
from brian2 import *
from scipy import spatial
import ot
from functools import partial
import logging
import sys
from joblib import Parallel, delayed
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.colorbar').disabled = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#% Distance functions %#
#Various Functions (and internal functions) to measure distances between spike trains
#mostly focused on time-indepedent distances

#% Earth Movers Distance %#

def emd_pdist_spk(spike_trains, nD=1, temporal=False, njobs=1, **kwargs):
    """
    """
    #this is the earth movers distance between all pairs of spike trains
    #this function takes a list of spike trains and returns the emd between all pairs of spike trains
    cdmat = np.zeros((len(spike_trains),len(spike_trains)))
    for i, st1 in enumerate(spike_trains):
        print(f"Computing EMD between spike train {i} and others")
        if njobs > 1: #if we are using parallelization, we need to use the parallel version of the function
            cdmat[i,:] = Parallel(n_jobs=njobs)(delayed(emd_spk_isi_dist)(st1, st2, nD=nD, temporal=temporal, kwargs=kwargs) for st2 in spike_trains)
        else:
            for j, st2 in enumerate(spike_trains):
                logger.debug(f"Computing EMD between spike train {i} and {j}")
                if i==j:
                    cdmat[i,j] = 0
                else:
                    cdmat[i,j] = emd_spk_isi_dist(st1, st2, nD=nD, temporal=temporal, kwargs=kwargs)
    
    return cdmat

def emd_pdist_isi(isi_trains, nD=1, temporal=False, **kwargs):
    pass

def emd_pdist_isi_hist(isi_hists, nD=1, temporal=False, **kwargs):
    """"""
    #this is the earth movers distance between all pairs of precomputed isi histograms
    #this function takes a list of isi histograms and returns the emd between all pairs of isi histograms
    cdmat = np.zeros((len(isi_hists),len(isi_hists)))
    for i, st1 in enumerate(isi_hists):
        for j, st2 in enumerate(isi_hists):
            logger.debug(f"Computing EMD between isi hist {i} and {j}")
            if i==j:
                cdmat[i,j] = 0
            else:
                cdmat[i,j] = emd_isi_hist(st1, st2)
    return cdmat

def emd_spk_isi_dist(spike_train1, spike_train2, nD=1, temporal=False, kwargs={} ):
    """ This is the earth movers distance between two spike trains, 
    this function takes the two spike trains and returns the emd between their isi distributions.
    Takes:

    """
    if nD == 1: #if we are in 1d, we can just use the emd function
        errorFunc = partial(emd_isi, **kwargs)
    elif nD >= 2: #if we are in 2d, we use the joint isi distribution
        if temporal:
            errorFunc = partial(temporal_isi_swasserstein_dd, spk1=spike_train1*1000, spk2=spike_train2*1000, n=nD, **kwargs)
        else:
            errorFunc = partial(isi_wasserstein_dd, n=nD, **kwargs)
    else:
        raise NotImplementedError("nD not implemented")
    
    isi1 = np.diff(spike_train1)*1000
    isi2 = np.diff(spike_train2)*1000
    dist = errorFunc(isi1=isi1, isi2=isi2)
    return dist

def isi_hist(isi1):
    bins = np.logspace(0,4)
    hisi1 = np.histogram(isi1, bins, density=False)[0].astype(np.float64)
    hisi1 /= hisi1.sum()
    if np.all(hisi1==0):
        hisi1 += 1e-6
    return hisi1

def emd_isi(isi1,isi2):
    bins = np.logspace(0,4)
    hisi1 = np.histogram(isi1, bins, density=False)[0].astype(np.float64)
    hisi2 = np.histogram(isi2, bins, density=False)[0].astype(np.float64)
    hisi1 /= hisi1.sum()
    hisi2 /= hisi2.sum()
    hisi1 = np.nan_to_num(hisi1, nan=1e-6)
    hisi2 = np.nan_to_num(hisi2, nan=1e-6)
    if np.all(hisi1==0):
        hisi1 += 1e-6
    if np.all(hisi2==0):
        hisi2 += 1e-6
    
    hisi1 = np.nan_to_num(hisi1, nan=1e-6)


    dist = stats.wasserstein_distance(np.arange(hisi1.shape[0]), np.arange(hisi1.shape[0]), hisi1, hisi2)

    return dist

def emd_isi_hist(isihist1, isihist2):
    if np.all(isihist1==0):
        isihist1 += 1e-6
    if np.all(isihist2==0):
        isihist2 += 1e-6
    dist = stats.wasserstein_distance(np.arange(len(isihist1)), np.arange(len(isihist2)), isihist1, isihist2)
    return dist

def isi_swasserstein_2d(isi1, isi2, bin=28, log=True, plot=False, savefig_name=''):
    
    xbins = np.linspace(0,4,bin+1)
    ybins = np.linspace(0,4,bin+1)
    if log:
        isi1, isi2 = np.log10(isi1+1), np.log10(isi2+1)
    isi_corr1, isi_corr2 = build_n_train(isi1), build_n_train(isi2)
    hist1, _, _ = np.histogram2d(isi_corr1[:,0], isi_corr1[:,1], bins=(xbins, ybins))
    hist2, _, out_bins = np.histogram2d(isi_corr2[:,0], isi_corr2[:,1], bins=(xbins, ybins))
    #print(hist1.max())
    #print(hist2.max())
    
    if np.all(hist1==0):
        hist1 += 0.5
    if np.all(hist2==0):
        hist2 += 0.5
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()
    dist = sliced_wasserstein(hist1, hist2, 500, bins=out_bins[:-1])

    if plot:
        plt.figure(num=920)
        plt.clf(),
        plot_xy(10**isi_corr2, fignum=920, color='k')
        plot_xy(10**isi_corr1, fignum=920, color='r')
        
        #plt.tight_layout()
        plt.savefig(savefig_name)

    return dist


def isi_wasserstein_dd(isi1, isi2, n=2, bin=28, log=True, norm=False, hist=True):
    """ Wasserstein distance between two spike trains. This function takes the two isi trains and returns the emd between their nD joint ISI distributions.
    Takes:
        isi1: isi train 1 (in ms)
        isi2: isi train 2 (in ms)
        n: dimension of the joint isi distribution
        bin: number of bins in the joint isi distribution
        log: whether to log transform the isi
        norm: whether to normalize the distance matrix
        hist: whether to use the histogramdd method or the raw ot method
    returns:
        dist: the wasserstein distance between the two spike trains
    """
    _bins = np.linspace(0,4,bin+1)
    if log:
        isi1, isi2 = np.log10(isi1+1), np.log10(isi2+1)
    isi_corr1, isi_corr2 = build_n_train(isi1, n=n), build_n_train(isi2, n=n)
    if hist:
        bins = [_bins for i in range(isi_corr1.shape[1])]
        hist1, _= np.histogramdd(isi_corr1, bins=bins)
        hist2, out_bins = np.histogramdd(isi_corr2, bins=bins)
        if np.all(hist1==0):
            hist1 += 0.5
        if np.all(hist2==0):
            hist2 += 0.5
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()
        
        #we need to calc the distance between the two distributions
        #on a nD grid, unfortunately ot.dist only works for 2d, so we need to flatten the distributions
        #and then reshape them back to the original shape
        #we also need to calc the distance matrix between the bins
        
        coords = flatten_coords([x[:-1] for x in bins])
        m_dist = ot.dist(coords,metric='euclidean')
        if norm:
            m_dist /= m_dist.sum()

        dist = ot.lp.emd2(hist1.flatten(), hist2.flatten(), m_dist)

    else:
        m_dist = ot.dist(isi_corr1, isi_corr2)
        if norm:
            m_dist /= m_dist.sum()
        dist = ot.lp.emd2([], [], m_dist)
    return dist

def temporal_isi_swasserstein_dd(spk1, spk2, isi1, isi2, n=2, tbin=100, bin=5, log=True, norm=True, hist=False):
    """ Temporally resolved wasserstein distance between two spike trains. This function takes the two spike trains and returns the emd between their joint ISI distributions.
    Takes:
        spk1: spike train 1 (in ms)
        spk2: spike train 2 (in ms)
        isi1: isi train 1 (in ms)
        isi2: isi train 2 (in ms)
        n: dimension of the joint isi distribution
        tbin: temporal bin size (in ms)
        bin: number of bins in the joint isi distribution
        log: whether to log transform the isi
        norm: whether to normalize the distance matrix
        hist: whether to use the histogramdd method or the raw ot method (default False; for temporally resolved)
    returns:
        dist: the wasserstein distance between the two spike trains
    """
    t_bins = np.arange(np.min(np.hstack((spk1, spk2))), np.max(np.hstack((spk1, spk2)))+tbin, tbin)
    _bins = np.linspace(0,4,bin+1)
    full_bins = [t_bins, *[_bins for i in range(n)]]
    if log:
        isi1, isi2 = np.log10(isi1 + 1), np.log10(isi2 + 1)
    isi_corr1, isi_corr2 = build_n_train(isi1, n=2), build_n_train(isi2,n=2)
    #stack the abs spike times back on to the isi, #paddding with 0s
    isi_corr1 = np.hstack((spk1[:-n].reshape(-1, 1), isi_corr1))
    isi_corr2 = np.hstack((spk2[:-n].reshape(-1, 1), isi_corr2))
    
    if hist:
        hist1, _ = np.histogramdd(isi_corr1, bins=full_bins)
        hist2, bins = np.histogramdd(isi_corr2, bins=full_bins)
        
        
        if np.all(hist1==0):
            hist1 += 0.5
        if np.all(hist2==0):
            hist2 += 0.5
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()
        
        #we need to calc the distance between the two distributions
        #on a nD grid, unfortunately ot.dist only works for 2d, so we need to flatten the distributions
        #and then reshape them back to the original shape
        #we also need to calc the distance matrix between the bins
        
        coords = flatten_coords([x[:-1] for x in bins])
        m_dist = ot.dist(coords, coords)
        m_dist /= m_dist.sum()
        dist = ot.lp.emd2(hist1.flatten(), hist2.flatten(), m_dist)
    else:
        #we can run the ot distance directly on the spike trains
        m_dist = ot.dist(isi_corr1, isi_corr2)
        if norm:
            m_dist /= m_dist.sum()
        dist = ot.lp.emd2([], [], m_dist)
    return dist

def biemd(isi1, isi2, lower=11, upper=200):
    """
    This is the biemd, which is the emd of the lower and upper parts of the isi, 
    here we are looking for the emd of the tonic and burst parts of the isi dist
    
    """
    #lower emd 
    lisi1 = isi1[isi1<=lower]
    lisi2 = isi2[isi2<=lower]
    lower_dist = emd_isi(lisi1, lisi2)
    #upper emd
    uisi1 = isi1[isi1>=upper]
    uisi2 = isi2[isi2>=upper]
    upper_dist = emd_isi(uisi1, uisi2)
    return lower_dist+upper_dist

#this is the sliced wasserstein distance for 2d distributions, can be superseded by the ot version
def sliced_wasserstein(X, Y, num_proj=500, bins=None):
    '''Takes:
        X: 2d (or nd) histogram normalized to sum to one
        Y: 2d (or nd) histogram normalized to sum to one
        num_proj: Number of random projections to compute the mean over
        ---
        returns:
        mean_emd_dist'''
     #% Implementation of the (non-generalized) sliced wasserstein (EMD) for 2d distributions as described here: https://arxiv.org/abs/1902.00434 %#
    # X and Y should be a 2d histogram 
    # Code adapted from stackoverflow user: Dougal - https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
    dim = X.shape[1]
    if bins is None:
        bins = np.arange(dim)
    ests = []
    for x in range(num_proj):
        # sample uniformly from the unit sphere
        dir = np.random.rand(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir

        # compute 1d wasserstein
        ests.append(stats.wasserstein_distance(bins, bins, X_proj, Y_proj))
    return np.mean(ests)

def flatten_coords(bins):
    #bins will be a list of bin edges, with possibly different bin sizes and array lengths. we need to generate a list of all coords
    grid = np.meshgrid(*bins)
    coords = np.vstack([x.flatten() for x in grid]).T
    return coords



#% Other Distance Functions %#

def kde_dist(kde1, kde2):
    dist = kde1[0].integrate_kde(kde2[0])
    norm2 = kde2[0].integrate_kde(kde2[0])
    norm1 = kde1[0].integrate_kde(kde1[0])
    dist = 1/(dist/ (norm1+norm2))
    return dist

def pdist_kde(list_kde):
    ar_kde = np.vstack(list_kde)
    cd_mat = spatial.distance.pdist(ar_kde, kde_dist)
    return cd_mat

def isi_kde_dist(isi1, isi2):
    #turn to 2d train
    isi_corr1, isi_corr2 = build_n_train(isi1), build_n_train(isi2)
    #build kdes
    kde1, kde2 = build_kde(isi_corr1), build_kde(isi_corr2)

    dist = kde_dist([kde1], [kde2])

    return dist

def ks_isi(isi1, isi2):
    dist = stats.ks_2samp(isi1, isi2)[0]
    return dist

def intra_burst_dist(isi1, isi2, low_cut=10, plot=False):
    intra_burst1, pre_burst1 = intra_burst_hist(isi1, low_cut=low_cut)
    intra_burst2, pre_burst2 = intra_burst_hist(isi2, low_cut=low_cut)
    bins = np.linspace(0,4)
    hisi1 = np.histogram(np.log10(pre_burst1), bins)[0].astype(np.float64)
    hisi2 = np.histogram(np.log10(pre_burst2), bins)[0].astype(np.float64)
    if hisi1.sum() < 1e-5:
        hisi1 += 0.5
    if hisi2.sum() < 1e-5:
        hisi2 += 0.5
    hisi1 /= hisi1.sum()
    hisi2 /= hisi2.sum()
    dist = stats.wasserstein_distance(np.arange(hisi1.shape[0]), np.arange(hisi2.shape[0]), hisi1, hisi2)
    
    if plot:
        plt.figure(565)
        plt.clf()
        bins = np.linspace(0,4)
        plt.hist(np.log10(pre_burst1), bins, alpha=0.25, density=True)
        plt.hist(np.log10(pre_burst2), bins, alpha=0.25, density=True)
        
    return dist

def tonic_dist(isi1, isi2, plot=False):
    intra_burst1, pre_burst1 = tonic_hist(isi1)
    intra_burst2, pre_burst2 = tonic_hist(isi2)
    bins = np.linspace(0,4)
    hisi1 = np.histogram(np.log10(pre_burst1), bins)[0].astype(np.float64)
    hisi2 = np.histogram(np.log10(pre_burst2), bins)[0].astype(np.float64)
    if hisi1.sum() < 1e-5:
        hisi1 += 0.5
    if hisi2.sum() < 1e-5:
        hisi2 += 0.5
    hisi1 /= hisi1.sum()
    hisi2 /= hisi2.sum()
    dist = stats.wasserstein_distance(np.arange(hisi1.shape[0]), np.arange(hisi2.shape[0]), hisi1, hisi2)
    
    if plot:
        plt.figure(565)
        plt.clf()
        bins = np.linspace(0,4)
        plt.hist(np.log10(pre_burst1), bins, alpha=0.25, density=True)
        plt.hist(np.log10(pre_burst2), bins, alpha=0.25, density=True)
        
    return dist

def enforce_no_burst(isi, threshold, print_index=False):
    burst_count = len(isi[isi<=8])
    overallsize = len(isi)
    burst_index = burst_count/overallsize
    if print_index:
        print(burst_index)
    if burst_index > threshold:
        return burst_index * 1000
    else:
        return burst_index

def enforce_burst(isi, threshold, print_index=False):
    burst_count = len(isi[isi<=8])
    overallsize = len(isi)
    burst_index = burst_count/overallsize
    if print_index:
        print(burst_index)
    if burst_index < threshold:
        return (1/(burst_index + 1e-4)) * 1000
    else:
        return (1/(burst_index + 1e-4))

def compute_burst_index(isi):
    burst_count = len(isi[isi<=8])
    overallsize = len(isi)
    burst_index = burst_count/overallsize
    return burst_index



