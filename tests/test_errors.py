from sNMO.error.spikeTrainErrors import emd_pdist_spk, isi_swasserstein_2d, isi_wasserstein_dd
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
#log debug to std out
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


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
    emd_matrix = emd_pdist_spk(spike_trains, nD=4, bin=8)
    assert np.all(emd_matrix[~np.eye(emd_matrix.shape[0],dtype=bool)] > 0)

        
    #Test case 6: temporal emd should be non-negative, on some random spike trains
    spike_trains = [np.cumsum(np.random.randint(0, 100, 100).astype(np.float32))/1000 for _ in range(10)] #in seconds
    emd_matrix = emd_pdist_spk(spike_trains, nD=2, temporal=True)
    assert np.all(emd_matrix[~np.eye(emd_matrix.shape[0],dtype=bool)] > 0)

    #Test case 6: 2d sliced and dd nD=2 should be roughly the same
    spike_trains = []
    spike_trains.append(np.cumsum(np.random.lognormal(6, 0.1, 200))) #in ms
    spike_trains.append(np.cumsum(np.random.lognormal(1, 0.1, 200))) #in ms
    emd = isi_swasserstein_2d(np.diff(spike_trains[0]), np.diff(spike_trains[1]), plot=True) #approx in euclidean space
    emd2 = isi_wasserstein_dd(np.diff(spike_trains[0]), np.diff(spike_trains[1]), norm=False) #approx in euclidean space
    assert np.isclose(emd, emd2, atol=0.9) #should be roughly the same
    #should show dots at the mean more or less
    plt.scatter(np.exp(6), np.exp(6))
    plt.scatter(np.exp(1), np.exp(1))
    plt.savefig('emd_matrix.png')
    assert emd > 0


if __name__ == "__main__":
    test_emd_pdist_spk()