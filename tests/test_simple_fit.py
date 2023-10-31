import sNMO
from sNMO.b2_model.brian2_model import brian2_model
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

def test_simple_fit():
    #create a simple dataset. In this case a LIF neuron
    model_1 = sNMO.b2_model.brian2_model.brian2_model(param_dict={'C': 90*pF, 'EL':-69*mV, 'taum': 100*ms, 'a': 3*nS, 'b': 100*pA , 'DeltaT': 1*mV, 'tauw': 100*ms, 'refrac': 2*ms, 'VT': -50*mV, 'VR': -60*mV, 'gL': 9*nS})
    
    #make some random stimulation
    stim = np.random.rand(30000)*20
    stim = stim.reshape(1, -1)
    #simulate the model
    model_1.add_real_data(np.arange(0, 1000, 0.1)*ms, np.zeros(stim.shape), stim, spike_times=np.arange(0, 1000, 100)*ms)
    model_1.activeSweep = 1
    model_1.dt = 0.1
    M,V = model_1.run_current_sim(sweepNumber=0)
    #plot the voltage
    plt.plot(V.t/second, V.v[0], label='voltage')
    plt.show()

if __name__ == '__main__':
    test_simple_fit()