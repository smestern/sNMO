from sNMO.error.spikeTrainErrors import emd_pdist_spk, isi_swasserstein_2d, isi_wasserstein_dd
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
from joblib import dump, load
import os
from brian2.units import second
import time
#log debug to std out
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#get current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

def test_emd_pdist_spk():
     
    # Test case 1: identical spike trains should have EMD of 0
    spike_trains = [np.array([1, 2, 3], dtype=np.float64), np.array([1, 2, 3], dtype=np.float64), np.array([1, 2, 3], dtype=np.float64)]
    emd_matrix = emd_pdist_spk(spike_trains)
    assert np.all(emd_matrix == 0)
    
    # Test case 2: Two spike trains with different number of spikes should have non-zero EMD
    spike_trains = [np.array([1, 2, 3], dtype=np.float64), np.array([1.5,2, 4.5, 20], dtype=np.float64)]
    emd_matrix = emd_pdist_spk(spike_trains, nD=1)
    assert emd_matrix[0, 1] > 0
    
    # Test case 3: EMD should be symmetric
    spike_trains = [np.array([1, 2, 3], dtype=np.float64), np.array([1, 2], dtype=np.float64)]
    emd_matrix = emd_pdist_spk(spike_trains, nD=1)
    assert emd_matrix[0, 1] == emd_matrix[1, 0]

    # Test case 4: EMD should be non-negative, on some random spike trains
    spike_trains = [np.cumsum(np.random.randint(0, 100, 100).astype(np.float32))/1000 for _ in range(10)] #in seconds
    emd_matrix = emd_pdist_spk(spike_trains, nD=2)
    assert np.all(emd_matrix[~np.eye(emd_matrix.shape[0],dtype=bool)] > 0)
    

    # Test case 5: EMD should function on nD=4, on some random spike trains
    spike_trains = [np.cumsum(np.random.randint(0, 100, 100).astype(np.float32))/1000 for _ in range(10)] #in seconds
    emd_matrix = emd_pdist_spk(spike_trains, nD=4, bins=8)
    assert np.all(emd_matrix[~np.eye(emd_matrix.shape[0],dtype=bool)] > 0)

        
    #Test case 6: temporal emd should be non-negative, on some random spike trains
    spike_trains = [np.cumsum(np.random.randint(0, 100, 100).astype(np.float32))/1000 for _ in range(10)] #in seconds
    emd_matrix = emd_pdist_spk(spike_trains, nD=2, temporal=True, hist=True)
    assert np.sum(emd_matrix) > 0

    #Test case 6: 2d sliced and dd nD=2 should be roughly the same
    spike_trains = []
    spike_trains.append(np.cumsum(np.random.lognormal(6, 0.1, 200))) #in ms
    spike_trains.append(np.cumsum(np.random.lognormal(1, 0.1, 200))) #in ms
    emd = isi_swasserstein_2d(np.diff(spike_trains[0]), np.diff(spike_trains[1])) #approx in euclidean space
    emd2 = isi_wasserstein_dd(np.diff(spike_trains[0]), np.diff(spike_trains[1]), norm=False, hist=True, bins=29, dist_metric='euclidean') #approx in euclidean space, 29 bins, normalized and histogrammed
    assert np.isclose(emd, emd2, atol=0.9) #should be roughly the same

    #test case 7: 2d sliced spikes from file.
    spike_trains = load(dir_path + '/demo_spikes_N2.joblib')[:50]
    emd = emd_pdist_spk(spike_trains, nD=2, njobs=6, hist=True, temporal=True)
    assert np.any(emd[~np.eye(emd.shape[0],dtype=bool)] > 0)
    

def test_emd_pdist_spk_with_kwargs():
    #emd_pdist_spk(spike_trains, nD=1, temporal=False, bin=1, **kwargs)
    #Test case 7: 2d sliced spikes from file.
    spike_trains = load(dir_path + '/demo_spikes_N.joblib')
    #convert to a list of numpy arrays
    spike_trains = [np.array(st/second) for st in spike_trains.values()][:500]
    emd_matrix = emd_pdist_spk(spike_trains, nD=2, temporal=False, hist=False, norm=True)
    assert np.any(emd_matrix[~np.eye(emd_matrix.shape[0],dtype=bool)] > 0)
    #the two matrix should correlate
    #time consuming?
    st_time = time.time()
    emd_matrix2 = emd_pdist_spk(spike_trains, nD=2, njobs=6, temporal=False, hist=True, norm=True, n_threads=6)
    logger.info('Elapsed time: %s', (time.time() - st_time)/60.0)
    assert np.corrcoef(emd_matrix.flatten(), emd_matrix2.flatten())[0,1] > 0.9


if __name__ == "__main__":
    test_emd_pdist_spk()
    test_emd_pdist_spk_with_kwargs()