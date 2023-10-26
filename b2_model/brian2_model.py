from itertools import filterfalse
from re import X
from utils import *
from brian2 import *
import b2_model.models as models
from scipy import stats
from ipfx import feature_extractor
from ipfx import subthresh_features as subt


class brian2_model(object):
    
    def __init__(self, model='adEx', param_dict=None):
        ## Default Values
        self.N = 1
        self._run_time = 3

        #load the choosen model 
        self._model =  getattr(models, model)
        #passed params
        self.record_vars = ['v']
        if param_dict is not None:
            self.__dict__.update(param_dict) #if the user passed in a param dictonary we can update the objects attributes using it.
                # This will overwrite the default values if the user passes in for example {'Cm:' 99} the cm value above will be overwritten

        #for optimization purposes, we can also figure out the optimiziable parameters without the user having to pass them in
        self._optimiziable_params = self._get_optimiziable_params()


    def run_current_sim(self, sweepNumber=None, param_dict=None):

        if param_dict is not None:
            self.__dict__.update(param_dict)
        if sweepNumber is not None and sweepNumber != self.activeSweep:
            self.activeSweep = sweepNumber
        
        start_scope()
        temp_in = self.realC[self.activeSweep,:]
        in_current = TimedArray(values = temp_in * pamp, dt=self.dt * ms)
        P = NeuronGroup( self.N, model=self._model['eqs'],  threshold=self._model['threshold'], reset=self._model['reset'], refractory=self._model['refractory'], method=self._model['method'] )
        
        var_dict = P.get_states(read_only_variables=False)
        res_dict = self.find_matching_properites(var_dict, self.__dict__)
        P.set_states(res_dict)

        #if the user initilizes the parameters
        if 'init_var' in self._model.keys():
            for key, val in self._model['init_var'].items():
                if isinstance(val, str):
                    P.set_states({key: P.get_states(vars=[val])[val]})
                elif isinstance(val, float) or isinstance(val, int):
                    P.set_states({key: val})

        # monitor
        M = SpikeMonitor( P )
        V = StateMonitor( P, self.record_vars, record=True, dt=self.dt * ms, when='start')
        run(self._run_time * second)
        return M,V

    def find_matching_properites(self, dict_1, dict_2):
        #get the keys
        keys1 = list(dict_1.keys())
        keys2 = list(dict_2.keys())
        #find the matching keys
        intersect = np.in1d(keys2, keys1)
        keys_to_use = np.array(keys2)[intersect]
        out_dict =  { x: dict_2[x] for x in keys_to_use }
        return out_dict


    def build_params_from_data(self):
        self.activeSweep = self.subthresholdSweep
        self.dt = compute_dt(self.realX)
        self.EL = compute_rmp(self.realY, self.realC)
        self.VT = compute_threshold(self.realX, self.realY, self.realC, self.spikeSweep)
        self.spike_times = self._detect_real_spikes()
        self._compute_real_fi()
        #self._compute_subthreshold_features(self.realX[0], self.realY[0], self.realC[0])

    def _detect_real_spikes(self):
        return detect_spike_times(self.realX, self.realY, self.realC)

    def add_real_data(self,realX, realY, realC, spike_times, subthresholdsweep=[0], spikesweep=[5]):
        self.realX = realX
        self.realY = realY
        self.realC = realC
        self.spike_times = spike_times ##Passed in spike_times
        self.subthresholdSweep = subthresholdsweep ##If the user passed in the subthreshold sweep they want to fit to
        self.spikeSweep = spikesweep
        ##If the user passed in the spiking sweep to fit to
        return
    
    def _compute_real_fi(self):
        self.bins = np.logspace(0, 3, 50)
        self.realFI, self.realISI = compute_FI_curve(self.spike_times, self._run_time)
        self.realISIHist =  compute_sweepwise_isi_hist(self.spike_times, self._run_time, self.bins)
        return

    #== Optimization functions ==

    def opt_spikes(self, param_dict, use_stored_spikes=False):
        self.__dict__.update(param_dict)
        error_s = np.zeros(self.N)
        
        for sweep in np.asarray(self.spikeSweep):
            self.activeSweep = sweep ##set the active sweep to the spiking sweep set by user
            spikes, _ = self.run_current_sim() ##Run the sim and grab the spikes
            self.temp_spike_array = spikes.spike_trains()
            sweep_error_s = []
            for unit in np.arange(0, self.N):
                temp = self.temp_spike_array[unit] / second
                if len(temp)< 1: 
                    temp_dist = 99999
                else:
                    try:
                        temp_dist = compute_spike_dist_euc(np.hstack((np.log10(temp[0]*1000), np.diff(temp)*1000)), np.hstack((np.log10(self.spike_times[self.activeSweep][0]*1000), np.diff(self.spike_times[self.activeSweep])*1000)))
                    except:
                        temp_dist = 99
                sweep_error_s = np.hstack((sweep_error_s, temp_dist))
            error_s = np.vstack((error_s, sweep_error_s))
        error_s = np.sum(error_s, axis=0)
        del self.temp_spike_array
        return np.nan_to_num(error_s, nan=999)

    def opt_trace(self, param_dict, during_stim=True):
        self.__dict__.update(param_dict)
        error_t = np.zeros(self.N)
        for sweep in np.asarray(self.subthresholdSweep):
            stim_t = find_stim_changes(self.realC[0,:])
            self.activeSweep = sweep
            _, traces = self.run_current_sim()
            if during_stim:
                
                sweep_error_t = np.apply_along_axis(compute_mse,1,(traces.v /mV)[:,stim_t[0]:stim_t[1]],self.realY[sweep,stim_t[0]:stim_t[1]])
            else:
                sweep_error_t = np.apply_along_axis(compute_mse,1,(traces.v /mV),self.realY[sweep,:])
            error_t = error_t + sweep_error_t
        error_t /= len(self.subthresholdSweep)
        error_t = np.nan_to_num(error_t, nan=9999, posinf=9999, neginf=9999)
        return error_t

    def opt_trace_and_spike_mse(self, param_dict):
        return self.opt_full_mse(param_dict)

    def opt_full_mse(self, param_dict):
        """_summary_

        Args:
            param_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.__dict__.update(param_dict) ##Apply the passed in params to the model
        if param_dict is not None:
            self.__dict__.update(param_dict)
        spikes_full = []
        isi_full = []
        spiking_vm = []
        error_t = np.zeros(self.N)
        error_s = np.zeros(self.N)
        neuronwise_isi_full = [[] for p in range(self.N)]
        for sweep in np.arange(self.realX.shape[0]):
            self.activeSweep = sweep ##set the active sweep to the spiking sweep set by user
            spikes, traces= self.run_current_sim() ##Run the sim and grab the spikes
            self.temp_spike_array = spikes.spike_trains()
            neuron_wise_spikes = []
            neuron_wise_isi = []
            neuron_wise_spiking_vm = [np.mean(traces.v[:]/mV, axis=1)]
            spiking_vm.append(neuron_wise_spiking_vm)
            sweep_error_s = []
            for p in np.arange(self.N):
                pspikes = self.temp_spike_array[p]
                if len(pspikes) > 0:
                    neuron_wise_spikes.append(len(pspikes))
                    spike_s = pspikes/ms
                    if len(spike_s) > 1:
                        neuronwise_isi_full[p].append(np.diff(spike_s))
                        neuron_wise_isi.append(emd_isi(np.append(np.diff(spike_s),1), np.append(np.diff(self.spike_times[self.activeSweep]*1000),1)))
                    else:
                        neuron_wise_isi.append(emd_isi(np.append([0],1), np.append(np.diff(self.spike_times[self.activeSweep]*1000),1)))
                    if sweep in self.spikeSweep:
                        if len(spike_s)< 1: 
                            temp_dist = np.nan
                        else:
                            try:
                                temp_dist = compute_spike_dist(spike_s/1000, self.spike_times[self.activeSweep])
                            except:
                                temp_dist = 99999
                        sweep_error_s.append(temp_dist)
                else:
                    sweep_error_s.append(np.nan)
                    neuron_wise_spikes.append(0)
                    neuron_wise_isi.append(emd_isi(np.append([0],1), np.append(np.diff(self.spike_times[self.activeSweep]*1000),1)))
            del self.temp_spike_array
            isi_full.append(np.nan_to_num(np.hstack((neuron_wise_isi))))
            spikes_full.append(np.hstack(neuron_wise_spikes))
            #if its subthreshold
            if sweep in self.subthresholdSweep:
                sweep_error_t = np.apply_along_axis(compute_mse,1,(traces.v /mV),self.realY[sweep,:])
                error_t = error_t + sweep_error_t
            if sweep in self.spikeSweep:     
               error_s = np.vstack((error_s, np.hstack(sweep_error_s)))
               temp_min = np.apply_along_axis(compute_min_stim, 1, (traces.v /mV), traces[0].t/second, strt=0.2, end=1.2)
               real_min = compute_min_stim(self.realY[sweep,:], self.realX[sweep,:], strt=0.2, end=1.2)
               sweep_error_t = np.apply_along_axis(compute_mse,1,temp_min.reshape(-1,1),real_min)
               error_t = error_t + sweep_error_t
        spiking_vm = np.vstack(spiking_vm)
        #Sum error_s and take out zeros
        error_s = np.nansum(error_s, axis=0)
        error_s[error_s==0.0] = 99999
        error_s /= len(self.spikeSweep)

        #compute the FI curves
        spikes_return = np.vstack(spikes_full)
        isi_return = np.vstack(isi_full)
        error_t /= (len(self.subthresholdSweep) + len(self.spikeSweep))
        spikes_return, isi_return = (spikes_return.T / self._run_time), isi_return.T
        neuron_wise_isi_hist = np.vstack([np.histogram(np.hstack((s)), bins=self.bins)[0] if len(s)>0 else np.zeros(len(self.bins)-1) for s in neuronwise_isi_full])
        unit_wise_isi_e = np.hstack([compute_emd_1d(self.realISIHist, row) for row in neuron_wise_isi_hist]) / 10
        unit_wise_error = np.apply_along_axis(compute_mse,1,spikes_return,self.realFI)
        error_fi = unit_wise_error
        error_isi = unit_wise_isi_e

        #Penalize the error fi if vm goes wayy to high
        spiking_vm = np.nan_to_num(spiking_vm, nan=9999, posinf=9999, neginf=999)
        max_vm = np.amax(spiking_vm, axis=0)
        bool_max = max_vm>150
        error_fi[bool_max] *= 100

        #take out the nan's
        error_t = np.nan_to_num(error_t, nan=9999)
        error_fi = np.nan_to_num(error_fi, nan=9999)
        
        return np.nanmean((error_t, error_fi, error_isi, error_s)), error_t, error_fi, error_isi, error_s

    def opt_FI(self,param_dict=None, log_scale=False):
        if param_dict is not None:
            self.__dict__.update(param_dict)
        unit_wise_FI, unit_wise_isi = self.build_FI_curve()
        real_FI = self.realFI
        if log_scale:
            log_sim = np.nan_to_num(np.log10(unit_wise_FI+1), posinf=0, neginf=0)
            log_real = np.nan_to_num(np.log10(real_FI+1), posinf=0, neginf=0)
            log_sim_isi = np.nan_to_num(np.log10(unit_wise_isi+1), posinf=0, neginf=0)
            log_real_isi = np.nan_to_num(np.log10(self.realISI+1), posinf=0, neginf=0)
            unit_wise_isi_e = np.apply_along_axis(compute_mlse,1,log_sim_isi,log_real_isi)
            unit_wise_error = np.apply_along_axis(compute_mlse,1,log_sim,log_real)
        else:
            unit_wise_isi_e = np.apply_along_axis(compute_mse,1,unit_wise_isi,self.realISI)
            unit_wise_error = np.apply_along_axis(compute_mse,1,unit_wise_FI, self.realFI)
        return unit_wise_error, unit_wise_isi_e

    def build_FI_curve(self, param_dict=None):
        if param_dict is not None:
            self.__dict__.update(param_dict)
        spikes_full = []
        isi_full = []
        for sweep in np.arange(self.realX.shape[0]):
            self.activeSweep = sweep ##set the active sweep to the spiking sweep set by user
            spikes, _ = self.run_current_sim() ##Run the sim and grab the spikes
            self.temp_spike_array = spikes.spike_trains()
            neuron_wise_spikes = []
            neuron_wise_isi = []
            for p in np.arange(self.N):
                pspikes = self.temp_spike_array[p]
                if len(pspikes) > 0:
                    neuron_wise_spikes.append(len(pspikes))
                    spike_s = pspikes/ms
                    if len(spike_s) > 1:
                        neuron_wise_isi.append(np.nanmean(np.diff(spike_s)))
                    else:
                        neuron_wise_isi.append(0)
                else:
                    neuron_wise_spikes.append(0)
                    neuron_wise_isi.append(0)
            isi_full.append(np.nan_to_num(np.hstack((neuron_wise_isi))))
            spikes_full.append(np.hstack(neuron_wise_spikes))

        spikes_return = np.vstack(spikes_full)
        isi_return = np.vstack(isi_full)
        del self.temp_spike_array
        return spikes_return.T / self._run_time, isi_return.T

    def build_feature_curve(self, param_dict):
        self.__dict__.update(param_dict) ##Apply the passed in params to the model
        if param_dict is not None:
            self.__dict__.update(param_dict)
        spikes_full = [[] for x in np.arange(self.N)]
        isi_full = [[] for x in np.arange(self.N)]
        ##Generate an empty list of length N_UNITS which allows us to dump the subthreshold params into
        subthres_features = [[] for x in np.arange(self.N)]
        rmp_full = [[] for x in np.arange(self.N)]
        stim_min = [[] for x in np.arange(self.N)]
        for i, sweep in enumerate(self.realC): #For each sweep
            print(f"Simulating sweep {i}")
            self.activeSweep = i ##set the active sweep to the spiking sweep set by user
            spikes, voltage = self.run_current_sim()
            temp_spike_array = spikes.spike_trains() # Grab the spikes oriented by neuron in the network
            print("Simulation Finished")
            for p in np.arange(self.N): #For each neuron

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
                    rmp_full[p].append(temp_rmp)
                    if i in self.subthresholdSweep:
                        temp_deflection = compute_steady_hyp(voltage[p].v/mV.reshape(1,-1), sweep.reshape(1,-1)) #compute the end
                        subthres_features[p].append(temp_deflection)

                        #compute Sweepwisemin
                        temp_min = compute_min_stim(voltage[p].v/mV, voltage[0].t/second, strt=0.2, end=1.2)
                        stim_min[p].append(temp_min)
                
        neuron_avgs = np.vstack([np.nanmean(np.vstack(x), axis=0) for x in rmp_full])
        #of the SubT features
        spikes_return = np.array(spikes_full) #Stack all the arrays together
        isi_return = np.array(isi_full) #Stack all the arrays together
        min_return = np.array(stim_min)
        sub_return = np.array(subthres_features)
        return_full = np.hstack((spikes_return / self._run_time, isi_return, neuron_avgs, sub_return, min_return))
        return return_full

    def _internal_spike_error(self, x):
          temperror = compute_spike_dist(np.asarray(self.temp_spike_array[x] / second), self.spike_times[self.activeSweep]) 
          return temperror
    
    def _modified_mse(self, y, yhat):
        y_min = np.amin(y)
        y_max = np.amax(y)
        low_thres =(1.5 * np.amin(yhat))
        high_thres = np.amax(yhat) + (-0.5 * np.amax(yhat))
        if (np.amin(y) < low_thres) or (np.amax(y) > high_thres):
             return 99999
        else:
            return compute_mse(y, yhat)
   
    def _compute_rmp(self, dataY, dataC):
        deflection = np.nonzero(dataC)[0][0] - 1
        rmp1 = np.mean(dataY[:deflection])
        return rmp1

    def set_params(self, param_dict):
        self.__dict__.update(param_dict)

    def get_params(self):
        return self.__dict__

    def run_current_sweep(self, sweepNumber, param_dict=None):
        if param_dict is not None:
            self.__dict__.update(param_dict)
        self.activeSweep = sweepNumber
        return self.run_current_sim()
    
    def update_matching_attributes(self, neuron_group):
        self_keys = self.__dict__.keys()
        neuron_keys = neuron_group.__dict__.keys()
        for x in self_keys:
            if x in neuron_keys:
                neuron_group.__dict__[x] = self.__dict__[x]
        return neuron_group
    
    def _get_optimiziable_params(self):
        #to do this we need to denovo construct a neuron group
        start_scope()
        ngrp = NeuronGroup(1, model=self._model['eqs'],  threshold=self._model['threshold'], reset=self._model['reset'], refractory=self._model['refractory'], method=self._model['method'] )
        variables = ngrp.get_states(read_only_variables=False)
        optimiziable_params = {}
        for key, val in variables.items():
            if isinstance(val, Quantity):
                optimiziable_params[key] = {'value': val, 'unit': val.dim}
            else:
                optimiziable_params[key] = {'value': val, 'unit': None}
        return optimiziable_params




