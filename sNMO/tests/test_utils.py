import matplotlib.pyplot as plt
from sNMO.utils.spike_train_utils import cast_backend_spk, binned_fr, binned_fr_pop, \
build_networkx_graph, build_isi_from_spike_train, bin_signal, ecv2, cv, elv, rolling_window
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

    # arr = load(os.path.join(file_path, 'spks.pkl'))
    # out = cast_backend_spk(arr)
    # assert np.isclose(out[0][0], arr[0][0]/second).all(); print("test case 14 passed")

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

def test_binned_isi():
    print("Testing binned_isi")

    spikes = load(os.path.join(file_path, 'demo_spikes_N.joblib'))
    isi = [np.diff(x/second) for _,x  in spikes.items() if _ < 500]
    bins = np.arange(0, 100, 0.1)
    #test case 1: bin_signal with a list of spike trains and None bin signal should return the isis binned
    out = [bin_signal(np.cumsum(i), i, bins=bins, bin_func=None) for i in isi if len(i) > 1]

    test_first = np.hstack(out[0][1]) #assuming we didnt lose any values this should work out to the same as the original isi[0]
    assert np.all(np.isclose(test_first, isi[0])); print("test case 1 passed")

def test_binned_cv():
    print("Testing binned_cv")
    spikes = load(os.path.join(file_path, 'demo_spikes_N.joblib'))
    isi = [np.diff(x/second) for _,x  in spikes.items() if _ < 500]
    bins = np.arange(0, 100, 10)
    #test case 1: bin_signal with a list of spike trains and None bin signal should return the isis binned
    out = [bin_signal(np.cumsum(i), i, bins=bins, bin_func=None) for i in isi if len(i) > 1]
    
    #now compute the cv for each bin
    outcv = [cv(i) for i in out[0][1]]
    outcv2 = [ecv2(i) for i in out[0][1]]
    outlv = [elv(i) for i in out[0][1]]
    outcv, outcv2, outlv = np.nan_to_num(outcv), np.nan_to_num(outcv2), np.nan_to_num(outlv)
    assert len(outcv) == len(out[0][1]); print("test case 1 passed")
    plt.figure()
    plt.scatter(np.cumsum(isi[0]), isi[0]*1000)
    plt.yscale('log')
    plt.twinx()
    plt.plot(bins, outcv2, label='ecv2')
    plt.plot(bins, outcv, label='cv')
    plt.plot(bins, outlv, label='elv')
    plt.legend()
    plt.show()
        

def test_misc():
    arr = load(os.path.join(file_path, './/demo_spikes_N4.joblib'))
    isi = build_isi_from_spike_train([arr])[0]
    assert isi.shape[0] == 37; print("test case 1 passed")

def test_rolling_window():
    #make a sine wave for testing
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    #test case 1: rolling window should return the correct shape
    x, out = rolling_window(x, y, 10)
    assert out.shape == (1000, 10); print("test case 1 passed")

    plt.plot(x, out.mean(axis=1))
    plt.plot(x, y)
    plt.pause(0.1)

def test_nx_func():
    print("Testing nx_func")
    #load the connmatrix
    connmatrix = load(os.path.join(file_path, 'demo_connmatrix.joblib'))
    #test case 1: build_networkx_graph should return a networkx graph
    g = build_networkx_graph(connmatrix["IE"], connmatrix["EI"])
    assert g.number_of_nodes() == 1000; print("test case 1 passed")
    assert g.number_of_edges() == len(connmatrix["IE"].T)+len(connmatrix["EI"].T); print("test case 2 passed")


if __name__ == '__main__':
    test_rolling_window()
    test_misc()
    test_binned_cv()
    test_binned_isi()
    #test_nx_func()
    test_check_backend_spk()
    test_binned_fr()