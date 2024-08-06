import matplotlib.pyplot as plt
from sNMO.utils.spike_train_utils import cast_backend_spk, binned_fr, binned_fr_pop, build_networkx_graph, build_isi_from_spike_train
import numpy as np
from brian2.units import second, ms
from brian2 import SpikeGeneratorGroup, SpikeMonitor, Network
from joblib import dump, load

import os
TEST_SPIKE_MON = True
file_path = os.path.dirname(os.path.realpath(__file__))
def test_check_backend_spk():
    print("Testing check_backend_spk")
    #test case 1: check_backend_spk should return the numpy array if it is already a numpy array
    arr = np.random.randint(0, 100, (1000, 100))
    assert np.isclose(cast_backend_spk(arr), arr).all(); print("test case 1 passed")

    #test case 2: check_backend_spk should return the numpy array if it is a dict
    arr = np.random.randint(0, 100, (1000, 100))
    arr_dict = {i: arr[i] for i in range(arr.shape[0])}
    out = np.array(cast_backend_spk(arr_dict))
    assert np.isclose(out, arr).all(); print("test case 2 passed")

    #test case 3: check_backend_spk should return unitless numpy array if it is a list of quantities
    arr = np.random.randint(0, 100, (1000, 100))
    arr_units = [i*second for i in arr]
    out = np.array(cast_backend_spk(arr_units))
    assert np.isclose(out, np.array(arr)).all(); print("test case 3 passed")

    #test case 4: check_backend_spk should convert the units to seconds if it is a list of quantities, and then remove the units
    arr = np.random.randint(0, 100, (1000, 100))
    arr = [i*ms for i in arr]
    out = cast_backend_spk(arr)
    assert np.isclose(out[0], arr[0]/second).all(); print("test case 4 passed")

    #test case 5: check_backend_spk should convert the units to seconds if it is a dict of quantities, and then remove the units
    arr = np.random.randint(0, 100, (1000, 100))
    arr_dict = {i: arr[i]*ms for i in range(arr.shape[0])}
    out = cast_backend_spk(arr_dict)
    assert np.isclose(out[0], arr[0]/1000).all(); print("test case 5 passed")


    if TEST_SPIKE_MON:
        #test case 6: check_backend_spk should convert the units to seconds if it is a SpikeMonitor
        spikes = SpikeGeneratorGroup(3, [0, 1, 2], [0, 1, 2]*ms)
        spike_mon = SpikeMonitor(spikes)
        net = Network(spikes, spike_mon)
        net.run(5*second, report='text')
        out = cast_backend_spk(spike_mon)
        assert np.isclose(out[1], 1/1000); print("test case 6 passed")
        out2 = cast_backend_spk(spike_mon.spike_trains())
        assert np.isclose(out2[1], 1/1000); print("test case 7 passed")
        assert np.isclose(out[1], out2[1]); print("test case 9 passed")
        #try the same with the list
        out3 =[cast_backend_spk(x) for _,x  in spike_mon.spike_trains().items()]
        assert np.isclose(out3[1][0], 1/1000); print("test case 8 passed")

    #test case 7: check_backend_spk should handle a 3 level nested list
    arr = np.random.randint(0, 100, (1000, 100))
    arr_units = [[i*ms for i in arr] for _ in range(3)]
    out = cast_backend_spk(arr_units)
    assert np.isclose(out[0][0], arr[0]/1000).all(); print("test case 10 passed")

    #test case 8: check_backend_spk should handle a a single level list
    arr = np.random.randint(0, 100, (100))
    arr_units = [arr*ms]
    out = cast_backend_spk(arr_units)
    assert np.isclose(out[0], arr/1000).all(); print("test case 11 passed")
    out = cast_backend_spk(arr)
    assert np.isclose(out[0], arr[0]).all(); print("test case 12 passed")

    arr = load(os.path.join(file_path, 'demo_spikes_N4.joblib'))
    out = cast_backend_spk(arr)
    assert np.isclose(out[0], arr[0]).all(); print("test case 13 passed")

    arr = load(os.path.join(file_path, 'spks.pkl'))
    out = cast_backend_spk(arr)
    assert np.isclose(out[0][0], arr[0][0]/second).all(); print("test case 14 passed")

def test_binned_fr():
    print("Testing binned_fr")
    if TEST_SPIKE_MON:
        #test case 1: check_backend_spk should convert the units to seconds if it is a SpikeMonitor
        spikes = SpikeGeneratorGroup(3, np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 2]), np.array([443, 454, 427, 263, 430,  34, 205,  80, 419,  49])*ms)
        spike_mon = SpikeMonitor(spikes)
        net = Network(spikes, spike_mon)
        net.run(5*second, report='text')
        out = [binned_fr(x, 1)[0] for _,x  in spike_mon.spike_trains().items() if _ < 500]
        assert np.isclose(out[1][1], 0); print("test case 1 passed")
        #what happens if pass a list of spike trains?
        out = binned_fr(cast_backend_spk(spike_mon), 1)[0]
        assert out.shape == (3, 2); print("test case 2 passed")

        #try binned fr_pop
        out = binned_fr_pop(spike_mon, 1)[0]
        assert out.shape == (2,); print("test case 3 passed")
    
    #test case 3: check_backend_spk and binned_fr should handle a a dict items
    spikes = load(os.path.join(file_path, 'demo_spikes_N.joblib'))
    out = [binned_fr(x, 600)[0] for _,x  in spikes.items() if _ < 500]
    assert np.isclose(out[1][1], 0); print("test case 3 passed")


def test_misc():
    arr = load(os.path.join(file_path, './/demo_spikes_N4.joblib'))
    isi = build_isi_from_spike_train([arr])[0]
    assert isi.shape[0] == 37; print("test case 1 passed")



def test_nx_func():
    print("Testing nx_func")
    #load the connmatrix
    connmatrix = load(os.path.join(file_path, 'demo_connmatrix.joblib'))
    #test case 1: build_networkx_graph should return a networkx graph
    g = build_networkx_graph(connmatrix["IE"], connmatrix["EI"])
    assert g.number_of_nodes() == 1000; print("test case 1 passed")
    assert g.number_of_edges() == len(connmatrix["IE"].T)+len(connmatrix["EI"].T); print("test case 2 passed")


if __name__ == '__main__':
    test_misc()
    #test_nx_func()
    test_check_backend_spk()
    test_binned_fr()