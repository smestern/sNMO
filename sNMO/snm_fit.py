'''
Single neuron param fit
Rough coding space of several (optimizer, random, grid search) methods for optimizing
params. Can be called from the command line:
todo - add a command line interface
    - add a GUI interface
    - add a GUI interface with a progress bar
    - make more object oriented
'''
import argparse
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) #trying to log brian2 errors
logger = logging.getLogger(__name__)
import multiprocessing
import os
import time
import warnings
from pickle import dump, load
from functools import partial

import pandas as pd
import torch
from brian2 import *
from joblib import dump, load
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi
from sbi.inference.base import infer
from scipy import stats

from b2_model.brian2_model import brian2_model
from optimizer import snmOptimizer
import loadNWB as lnwb
from utils import *
from error import zErrorMetric,weightedErrorMetric 
warnings.filterwarnings("ignore")

#to allow parallel processing
prefs.codegen.target = 'cython'  # weave is not multiprocess-safe!
prefs.codegen.runtime.cython.multiprocess_safe = False
default_dtype = torch.float32

# === Global Settings ===
DEFAULT_Optimizer = partial(snmOptimizer, backend='ng')
ERROR_func = error_curves #should accept 2 arrays, the ground truth and the model output, and optionally object. Should return a N x 1 array of errors
ERROR_scaler = weightedErrorMetric

# === MAIN FUNCTION ===
class snmFitter():
    #WIP class that transforms the functional code below into a OOP class
    #this class should accept or create 3 objects, a model, a optimizer, and error metric
    #the model should be a brian2 model object, with a dataC, dataY, dataX, spike_time, and spikeSweep, 
    #   or if the user passes in a file, it should load the data and create the model based on optimizer_settings
    #the optimizer should be a snmOptimizer object, or a string that can be used to create a snmOptimizer object
    #the error metric should be a zErrorMetric object, or a function
    #the class should have a run method that runs the optimizer and returns the results

    def __init__(self, optimizer_settings=None, model=None, file=None, optimizer=DEFAULT_Optimizer, rounds=500, batch_size=500,
                 error_func=ERROR_func, error_scaler=ERROR_scaler, output_folder=None):

        #if the user passed in a file, load the data and model
        #or if the user passed in a model, use that model
        self._get_or_init_model_and_file(file, model, optimizer_settings)
        self.optimizer_settings = optimizer_settings
        
        self.rounds = rounds
        self.batch_size = batch_size
        self.output_folder = output_folder

        #if the optimizer is a string, create the optimizer object from the string
        self._get_or_init_optimizer(optimizer, optimizer_settings)
        
        #alias run as the run_optimizer function
        self.run = self.run_optimizer

        #handle the 

    def run_optimizer(self, file=None, optimizer_settings=None, optimizer=None, kwargs=None) -> dict:
        if kwargs is not None:
            #update the locals with the kwargs if the user passed them in
            if kwargs is not None:
                for key, value in kwargs.items():
                    if key in self.__dict__.keys():
                        self.__dict__[key] = value
        
        #if the user passed in a file, load the data and model
        if file is not None and file != self.file:
            self.model = self._load_data_and_model(file, optimizer_settings)
        
        #if the user passed in a optimizer, update the optimizer
        if optimizer is not None and optimizer != self.optimizer:
            self._get_or_init_optimizer(optimizer, optimizer_settings)

        
        return self.optimize(self.model, optimizer_settings, optimizer=optimizer)
    
    def optimize(self, model, optimizer_settings, optimizer=None) -> dict:
        """ Runs the specified optimizer over the file using the given settings. 
        Performs the basic optimization loop of asking for points, running them, and telling the output

        Args:
            model (b2_model): a brian2 model object containing a model and the ground truth data to be fit to.
            optimizer_settings (dict): a dict containg the optimizer settings. Specifically containg the constraints on the model varaiables 
            optimizer (str, optional): the optimization backend to be used. Can be one of ['ng', 'skopt', 'ax', 'sbi'] Defaults to 'ng'.
            id (str, optional): [description]. Defaults to 'nan'.

        Returns:
            results (dict): a dict containg the best fit parameters
        """
       
        model.set_params({'N': _batch_size})
        budget = int(_rounds * _batch_size)
        
        x_o = model_feature_curve(model)
        error_scaler = weightedErrorMetric(y=x_o, weights=[0.1, 1e-9, 1], splits=[[0, (len(model.spikeSweep)+len(model.subthresholdSweep))], [(len(model.spikeSweep)+len(model.subthresholdSweep)), (len(model.spikeSweep)+len(model.subthresholdSweep))*2], [(len(model.spikeSweep)+len(model.subthresholdSweep))*2, len(x_o)]])
        opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']].copy(), _batch_size, _rounds, 
        backend=optimizer)
        min_ar = []
        print(f"== Starting Optimizer with {_rounds} _rounds ===")
        for i in np.arange(_rounds):
            
        
            y = error_scaler.transform(np_o)

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
            print(f"[CELL {id}] - iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)} #with a min trace error {error_t[np.argmin(y)]} and FI error of {error_fi[np.argmin(y)]} and spike error of {error_isi[np.argmin(y)]}") #    
        results = opt.get_result() #returns a result containing the param - error matches
        return results

    def _load_data_and_model(self, file, optimizer_settings):
        """ loads a NWB file from a file path. Also computes basic parameters such as capactiance, if possible, from the data

        Args:
            file (str): a string pointing to a file to be loaded
            optimizer_settings (dict): a dict containing the optimizer settings. Specifically containing the constraints on the model varaiables
        """
        sweeps_to_use = optimizer_settings['sweeps_to_fit']
        file_path = file
        #set the stims 
        lnwb.global_stim_names.stim_inc = optimizer_settings['stim_names_inc']
        lnwb.global_stim_names.stim_exc = optimizer_settings['stim_names_exc']

        #load the data from the file
        dataX, dataY, dataC,obj = lnwb.loadFile(file_path, return_obj=True, old=False)

        #drop down to only the sweeps we want
        if sweeps_to_use != None:
            #load -70, -50, 50 70
            idx_stim = np.argmin(np.abs(dataX - 1))
            current_out = dataC[:, idx_stim]
            #take only the current steps we want
            current_idx = [x in sweeps_to_use for x in current_out]
            if np.sum(current_idx) < len(sweeps_to_use):
                raise ValueError("The file did not have the requested sweeps")
            else:
                dataX, dataY, dataC = dataX[current_idx,:], dataY[current_idx,:], dataC[current_idx,:]
            
        #if the settings say to run qc, run qc
        if optimizer_settings['run_qc']:
            dataX, dataY, dataC = sweepwise_qc(dataX, dataY, dataC)


        #detect the spike times,
        spike_time = detect_spike_times(dataX, dataY, dataC) 
        spiking_sweeps = np.nonzero([len(x) for x in spike_time])[0]
        non_spiking_sweeps = np.delete(np.arange(0, dataX.shape[0]), spiking_sweeps)

        #Create the model and attach the data. 
        model = brian2_model(model=optimizer_settings['model_choice'], param_dict={'dt':compute_dt(dataX), '_run_time':2, 'id': os.path.basename(file_path)})
        model.add_real_data(dataX, dataY, dataC, spike_time, non_spiking_sweeps, spiking_sweeps)

        #if the data is a square pulse, we can precompute tau, C, and el, and use those as constraints, to speed things up
        if "square" in optimizer_settings['stim_patt']:
            logger.debug("Square pulse detected, precomputing tau, C, and EL")
            #compute the tau, C, and R
            Resist = membrane_resistance_subt(dataX[non_spiking_sweeps], dataY[non_spiking_sweeps], dataC[non_spiking_sweeps])
            taum = np.nanmean([exp_decay_factor(dataX[x], dataY[x], dataC[x], plot=True) for x in non_spiking_sweeps])
            Capac = (mem_cap((Resist*Gohm)/ohm, taum) * farad) / pF
            #double check that taum or capacitance is not nan or too high
            param_pass = True
            if np.isnan(taum) or np.isnan(Capac):
                param_pass = False
            elif np.logical_and(taum <= 0.9, taum < 0.001):#taum is in seconds, so 900ms and 1ms
                param_pass = False
            elif np.logical_and(Capac <= 500, Capac <1):#capacitance is in pF, so 900pF and 1pF
                param_pass = False
            #if the params pass, set the constraints
            if param_pass:
                #We can constrain the C, R, and EL to be within 25% of the mean, to speed up the optimizer
                optimizer_settings['constraints'][optimizer_settings['model_choice']]['taum'] = [taum*0.75, taum*1.25]
                optimizer_settings['constraints'][optimizer_settings['model_choice']]['C'] = [Capac*0.75, Capac*1.25]
                optimizer_settings['constraints'][optimizer_settings['model_choice']]['R'] = [Resist*0.75, Resist*1.25]
                optimizer_settings['constraints'][optimizer_settings['model_choice']]['EL'] = [np.nanmean(dataY[:,0])*0.99, np.nanmean(dataY[:,0])*1.01]
        else:
            logger.debug("Non-square pulse detected, not precomputing tau, C, and EL")

   
    def _get_or_init_optimizer(self, optimizer, optimizer_settings):
        if self.optimizer_settings is None and optimizer_settings is None:
            logger.debug("No optimizer settings passed in, using default settings")
            self.optimizer = None
        elif isinstance(optimizer, str):
            self.optimizer = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']], self.batch_size, self.rounds, backend=optimizer)
        elif isinstance(optimizer, functools.partial): #if the optimizer is a partial, create the optimizer object
            self.optimizer = optimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']], self.batch_size, self.rounds)
        return self.optimizer
    
    def _get_or_init_model_and_file(self, file, model, optimizer_settings):
        #the user can pass in a file, or a model, but not both
        if file is not None and model is not None:
            raise ValueError("You cannot pass in both a file and a model")
        elif file is None and model is None and optimizer_settings is None:
            #if neither is passed in, thats okay, we just hope that the user passes it in later
            self.model = None
            self.file = None
        elif file is not None and optimizer_settings is not None: #if the user passed in a file and optimizer settings, load the data and model
            self.model = self._load_data_and_model(file, optimizer_settings)
            self.file = file
        elif file is None and optimizer_settings is not None: #if the user passed in just the optimizer settings, just spawn a model
            self.model =  brian2_model(model=optimizer_settings['model_choice'])
        elif file is not None:
            self.model = load_data_and_model(file, optimizer_settings)
            self.file = file
        elif model is not None: 
            self.model = model
            self.file = file
        return self.model, self.file


# === deprecated functional code below here ===


def run_optimizer(file, optimizer_settings, optimizer='ng', rounds=500, batch_size=500, sweep_upper_cut=None):
    ''' Runs the optimizer for a given file, using user specfied params and _rounds

    Takes:
    file (str) : A file path to a NWB file. To be fit to  
    (optional)  
    optimizer (str) : the optimizer protocol to use, either 'ng' or 'snpe'  
    rounds_ (int) : the number of _rounds to run  
    batch_size_ (int) : the number of neuron-param combos to try in parallel  

    Returns:
    results : the best fit params for the cell  
     '''
    global _rounds
    global _batch_size
    _rounds= rounds #Set the global settings to the user passed in params
    _batch_size= batch_size
    model = load_data_and_model(file, optimizer_settings, sweep_upper_cut=sweep_upper_cut) #load the 1nwb and model
    cell_id = os.path.basename(file) #grab the cell id by cutting around the file path
    #adjust the constraints to the found CM or TAUM
    optimizer_settings['constraints'][optimizer_settings['model_choice']]['EL'] = [model.EL*1.01, model.EL*0.99]
    optimizer_settings['constraints'][optimizer_settings['model_choice']]['C'] = [model.C*0.75, model.C*1.25]
    optimizer_settings['constraints'][optimizer_settings['model_choice']]['taum'] = [model.taum*0.75, model.taum*1.25]
    print(f"== Loaded cell {cell_id} for fitting ==")
    if optimizer == 'skopt' or optimizer=='ng' or optimizer=='ax':
        results = optimize(model, optimizer_settings, optimizer=optimizer, id=cell_id)
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

def load_data_and_model(file, optimizer_settings, sweep_upper_cut=None):
    ''' loads a NWB file from a file path. Also computes basic parameters such as capactiance   
    takes:  
    file (str): a string pointing to a file to be loaded  
    sweep_upper_cut (int): the sweep at which to stop indexing, sometimes useful to drop low quality sweeps  

    returns:  
    model (b2model): a brian2 model object with the real (in Vitro) data added  
    '''
    
  

    sweeps_to_use = optimizer_settings['sweeps_to_fit']
    file_path = file
    #set the stims
    lnwb.global_stim_names.stim_inc =  optimizer_settings['stim_names_inc']
    lnwb.global_stim_names.stim_exc = optimizer_settings['stim_names_exc']

    #load the data from the file
    cell_id = os.path.basename(file) #grab the cell id by cutting around the file path
    dataX, dataY, dataC,_ = lnwb.loadFile(file_path, old=False)
    
    #crop the data
    index_3 = np.argmin(np.abs(dataX[0,:]-2.50))
    ind_strt = np.argmin(np.abs((dataX[0,:]-0.50)))

    if sweep_upper_cut != None:
        dataX, dataY, dataC = dataX[:,ind_strt:index_3], dataY[:,ind_strt:index_3], dataC[:,ind_strt:index_3]
    elif sweeps_to_use != None:
        #load -70, -50, 50 70
        idx_stim = np.argmin(np.abs(dataX - 1))
        current_out = dataC[:, idx_stim]
        #take only the current steps we want
        current_idx = [x in sweeps_to_use for x in current_out]
        if np.sum(current_idx) < len(sweeps_to_use):
            raise ValueError("The file did not have the requested sweeps")
        else:
            dataX, dataY, dataC = dataX[current_idx,ind_strt:index_3], dataY[current_idx,ind_strt:index_3], dataC[current_idx,ind_strt:index_3]
    else:   
        dataX, dataY, dataC = dataX[:sweep_upper_cut,ind_strt:index_3], dataY[:sweep_upper_cut,ind_strt:index_3], dataC[:sweep_upper_cut,ind_strt:index_3]
    dataX = dataX - dataX[0,0]
    dataX, dataY, dataC = sweepwise_qc(dataX, dataY, dataC)
    sweeplim = np.arange(dataX.shape[0])
    dt = compute_dt(dataX)
    compute_el = compute_rmp(dataY[:2,:], dataC[:2,:])

    #baseline the data to the first two sweeps?
    sweepwise_el = np.array([compute_rmp(dataY[x,:].reshape(1,-1), dataC[x,:].reshape(1,-1)) for x in np.arange(dataX.shape[0])])
    sweep_offset = (sweepwise_el - compute_el).reshape(-1,1)
    dataY = dataY - sweep_offset

    #Compute Spike Times
    spike_time = detect_spike_times(dataX, dataY, dataC, sweeplim, upper=1.15) #finds geh 
    spiking_sweeps = np.nonzero([len(x) for x in spike_time])[0]
    rheobase = spiking_sweeps[0]
    non_spiking_sweeps = np.delete(np.arange(0, dataX.shape[0]), spiking_sweeps)
    thres = compute_threshold(dataX, dataY, dataC, sweeplim)

    #negative current sweeps 
    neg_current = [x<0 for x in dataC[:, np.argmin(np.abs(dataX-0.5))]]
    neg_current = np.arange(0, dataX.shape[0])[neg_current]

    #Compute cell params
    resistance = 1#membrane_resistance_subt(dataX[neg_current], dataY[neg_current], dataC[neg_current])
    taum = np.nanmean([exp_decay_factor(dataX[x], dataY[x], dataC[x], plot=True) for x in non_spiking_sweeps])
    plt.title(f"{taum*1000} ms taum")
    plt.savefig(f"output//{cell_id}_taum_fit.png")
    Cm = (mem_cap((resistance*Gohm)/ohm, taum)) * farad
    Cm = Cm/pF
    taum *=1000

    #Create the model and attach the data. 
    model = brian2_model(model=optimizer_settings['model_choice'], param_dict={'EL': compute_el, 'dt':dt, '_run_time':2, 'C': Cm, 'taum': taum, 'id': cell_id})
    model.add_real_data(dataX, dataY, dataC, spike_time, non_spiking_sweeps, spiking_sweeps)
    model.build_params_from_data()


    return model

# === fitting functions ===
## TODO These should be moved to a separate file, potentially a class

def SNPE_OPT(model, optimizer_settings, id='nan', run_ng=True, run_ng_phase=False, run_skopt=False):
    ''' Samples from a SNPE posterior to guess the best possible params for the data. Optionally runs the NEVERGRAD differential
    evolution optimizer restricted to the SNPE-generated top 100 most probable parameter space following sample.
    takes:
    model (a brian2_model object): with the cell data loaded into the objects properties
    id (str): a string containing the cell id. For saving the fit results
    use_post (bool): whether to use the prefit postierior (defaults to true)
    refit_post (bool): whether to refit the prefit postierior (defaults to false)
    run_ng (bool): whether to run a few _rounds of the optimizer after sampling from SNPE (defaults to True)
    '''
    from pickle import dump, load

    import torch
    from sbi import utils as utils
    from sbi.inference import SNLE, SNPE
    from sbi.inference.base import infer

    global dataC
    global dt
    global non_spiking_sweeps
    global N
    # Generate the X_o (observation) uses a combination of different params
    real_fi, real_isi = compute_FI_curve(model.spike_times, model._run_time) #Compute the real FI curve
    real_fi = np.hstack((real_fi, real_isi))
    real_rmp = compute_rmp(model.dataY, model.dataC)
    real_min = []
    real_subt = []
    for x in  model.subthresholdSweep :
        temp = compute_steady_hyp(model.dataY[x, :].reshape(1,-1), model.dataC[x, :].reshape(1,-1))
        temp_min = compute_min_stim(model.dataY[x, :], model.dataX[x,:], strt=0.62, end=1.2)
        real_subt.append(temp)
        real_min.append(temp_min)
    real_subt = np.array(real_subt)        
    
    real_min = np.hstack(real_min)
    N = _batch_size
    np_o = np.hstack((real_fi, real_rmp, real_subt, real_min))
    x_o = torch.tensor(np.hstack((real_fi, real_rmp, real_subt, real_min)), dtype=torch.float32)
    
    

    #model.subthresholdSweep = None
    opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']], _batch_size, _rounds, 
    backend='sbi', sbi_kwargs=dict(x_obs=x_o, prefit_posterior='ADIF_posterior_post.pkl')) #, sample_conditional={'C': model.C, 'taum': model.taum, 'EL': model.EL}
    #set the default X, seems to speed up sampling
    opt.rounds = 150
    opt.fit(model, id=model.id)
    # get top 100 samples
     ##TODO maybe take the mode of the highest samples for each column?
    #Take the 
    
    samples = opt.ask(n_points=500)
    log_prob = opt.posts[-1].log_prob(torch.tensor(opt.param_list.astype(np.float32)), norm_posterior=False).numpy()
    
    top_100 = opt.param_list[np.argsort(log_prob[-10:])]
   
    #Now run a few _rounds of optimizer over the data
    high = np.apply_along_axis(np.amax,0, top_100) #take the high range of each column (each column is a different parameter)
    low = np.apply_along_axis(np.amin,0, top_100) #take the low range
    var_pairs = np.transpose(np.vstack((low, high))) #stack them into low->pairs for each row
    for i, (key, val) in enumerate(optimizer_settings['constraints'][optimizer_settings['model_choice']].items()):
        if key != 'units':
            optimizer_settings['constraints'][optimizer_settings['model_choice']][key] = [var_pairs[i][0] - 1e-14, var_pairs[i][1] + 1e-14]

    if run_ng:
        results_out = optimize(model, optimizer_settings, id=model.id)  #returns a result containing the param - error matches
    elif run_ng_phase:
        results_out = biphase_opt(model, optimizer_settings)
    
    return results_out

def optimize(model, optimizer_settings, optimizer='ng', id='nan'):
    """ Runs the specified optimizer over the file using the given settings. 
    Performs the basic optimization loop of asking for points, running them, and telling the output

    Args:
        model (b2_model): a brian2 model object containing a model and the ground truth data to be fit to.
        optimizer_settings (dict): a dict containg the optimizer settings. Specifically containg the constraints on the model varaiables 
        optimizer (str, optional): the optimization backend to be used. Can be one of ['ng', 'skopt', 'ax', 'sbi'] Defaults to 'ng'.
        id (str, optional): [description]. Defaults to 'nan'.

    Returns:
        results (dict): a dict containg the best fit parameters
    """
    import nevergrad as ng


    model.set_params({'N': _batch_size})
    budget = int(_rounds * _batch_size)
    
    x_o = model_feature_curve(model)
    error_scaler = weightedErrorMetric(y=x_o, weights=[0.1, 1e-9, 1], splits=[[0, (len(model.spikeSweep)+len(model.subthresholdSweep))], [(len(model.spikeSweep)+len(model.subthresholdSweep)), (len(model.spikeSweep)+len(model.subthresholdSweep))*2], [(len(model.spikeSweep)+len(model.subthresholdSweep))*2, len(x_o)]])
    opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']].copy(), _batch_size, _rounds, 
    backend=optimizer, nevergrad_opt=ng.optimizers.ParaPortfolio)
    min_ar = []
    print(f"== Starting Optimizer with {_rounds} _rounds ===")
    for i in np.arange(_rounds):
        print(f"[CELL {id}] - iter {i} start")
        model.set_params({'N': _batch_size, 'refractory':0})
        t_start = time.time()
        param_list = opt.ask()
        print(f"sim {(time.time()-t_start)/60} min start")
        _, error_t, error_fi, error_isi, error_s = 0, np.zeros(_batch_size), np.zeros(_batch_size), np.zeros(_batch_size), np.zeros(_batch_size)#model.opt_full_mse(param_list)
        np_o = model.build_feature_curve(param_list)
        #error_fi = np.nan_to_num(error_fi, nan=999999) 
        
        #error_t  = np.nan_to_num(error_t , nan=999999, posinf=99999, neginf=99999)
        #y = np.abs(np.apply_along_axis(np.subtract, 1, np_o, x_o))  #error_fi + error_t

        #scale the error
       
        y = error_scaler.transform(np_o)

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
        print(f"[CELL {id}] - iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)} #with a min trace error {error_t[np.argmin(y)]} and FI error of {error_fi[np.argmin(y)]} and spike error of {error_isi[np.argmin(y)]}") #    
    results = opt.get_result() #returns a result containing the param - error matches
    return results

def biphase_opt(model, optimizer_settings, optimizer='ng', id='nan'):
    """Only tunes the AdEx model as it stands. First tunes the the subthreshold sweep, then the 

    Args:
        model ([type]): [description]
        optimizer_settings ([type]): [description]
        optimizer (str, optional): [description]. Defaults to 'ng'.
        id (str, optional): [description]. Defaults to 'nan'.

    Returns:
        [type]: [description]
    """
        
        
    import nevergrad as ng

    import copy as cpy
   
    model.set_params({'N': _batch_size})
    budget = int(_rounds * _batch_size)
    
    # subt params
    param_ranges = cpy.deepcopy(optimizer_settings['constraints'][optimizer_settings['model_choice']])
    #drop spiking params
    param_ranges.pop('VT')
    param_ranges.pop('b')
    param_ranges.pop('refrac')
    model.set_params({"VT":999999*mV}) #set the VT to a high value so it doesn't affect the fit

    param_ranges['units'].pop(3)
    param_ranges['units'].pop(5)
    param_ranges['units'].pop(7)
    opt = snmOptimizer(param_ranges, _batch_size, _rounds, backend=optimizer, nevergrad_opt=ng.optimizers.ParaPortfolio)
    min_ar = []
    min_points = []
    print(f"== Starting Optimizer with {_rounds} _rounds ===")
    for i in np.arange(_rounds):
        print(f"iter {i} start")
        model.set_params({'N': _batch_size, 'refractory':0, 'VT': 0*mV})
        t_start = time.time()
        param_list = opt.ask()
        print(f"sim {(time.time()-t_start)/60} min start")
        error_t = model.opt_trace(param_list)
        y = np.nan_to_num(error_t, nan=9999) 
        print(f"sim {(time.time()-t_start)/60} min end")
        opt.tell(param_list, y) ##Tells the optimizer the param - error pairs so it can learn
        t_end = time.time()
        min_ar.append(np.sort(y)[:5])
        min_points.append([opt.param_list[x] for x in np.argsort(y)[:5].tolist()])
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
    min_points_new = [] 
    for x in min_points:
        for params in x:
            min_points_new.append(params)
    min_ar = np.ravel(min_ar)
    res_min = pd.DataFrame.from_dict(min_points_new).to_dict('list')
    top_a_val = np.array(res_min['a'])[np.argsort(min_ar)[:100]]
    top_tauw_val = np.array(res_min['tauw'])[np.argsort(min_ar)[:100]]
    top_c_val = np.array(res_min['C'])[np.argsort(min_ar)[:100]]
    top_taum_val = np.array(res_min['taum'])[np.argsort(min_ar)[:100]]

    cm_m_ratio = top_c_val/top_taum_val
    _, cm_low, cm_high = mean_confidence_interval(cm_m_ratio)
    _ratio = top_a_val/top_tauw_val
    _, low_ratio, high_ratio = mean_confidence_interval(_ratio)
    mean_ratio = np.nanmean(_ratio)
    #Now intialize for F-I fitting
    #fix to the best fit values
    model.set_params(results)
    model.set_params({'N': _batch_size})
    model = sub_eq(model, 'a', f'a = (tauw/ms * {_})*nS : siemens')
    #create the cheap constraint
    def preserve_ratio(x):
        ratio = x['a'] / x['tauw'] 
        return (ratio <= high_ratio and ratio >= low_ratio)
    
    budget = int(_rounds * _batch_size) 


    param_ranges = cpy.deepcopy(optimizer_settings['constraints'][optimizer_settings['model_choice']])
    _, param_ranges['C'][0], param_ranges['C'][1] = mean_confidence_interval(top_c_val)
    _, param_ranges['taum'][0], param_ranges['taum'][1] = mean_confidence_interval(top_taum_val)

    opt = snmOptimizer(param_ranges, _batch_size, _rounds ** 20, backend=optimizer, nevergrad_opt=ng.optimizers.Portfolio)
    #opt.opt.parametrization.register_cheap_constraint(preserve_ratio)
    min_ar = []
    for i in np.arange(_rounds):
        print(f"iter {i} start")
        model.set_params({'N': _batch_size, 'refractory':0})
        t_start = time.time()
        param_list = opt.ask()
        print(f"sim {(time.time()-t_start)/60} min start")
        _, error_t, error_fi, error_isi, error_s = model.opt_full_mse(param_list)
        y = error_fi + error_s + (error_t/1000)
        print(f"sim {(time.time()-t_start)/60} min end")
        opt.tell(param_list, y) ##Tells the optimizer the par0am - error pairs so it can learn
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
        if len(min_ar) > 10:
                if _check_min_loss_gradient(min_ar, num_no_improvement=10) == False:
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
def _check_min_loss_gradient(min_ar, num_no_improvement=10, threshold=1e-5):
    """ Checks the minimum loss gradient for early stopping purposes. 
    Using a simple linear regression of the minimum found in each local round.

    Args:
        min_ar (numpy array): an array containing the found minimum for each round
        num_no_improvement (int, optional): Number of _rounds as a minimum for early stopping. Defaults to 10.
        threshold (float, optional): The minimum slope (in either direction) for the _rounds to be stopped. Defaults to 1e-5.

    Returns:
        pass_check (bool): boolean indicating whether the 
    """
    pass_check = True
    min_ar = np.array(min_ar)
    min_ar = np.nanmean(min_ar, axis=1)
    if min_ar.shape[0] <= num_no_improvement:
        logging.debug(f"Not enough rounds to check for minimum loss gradient")
        pass
    else:
        x = np.arange(0, num_no_improvement)
        slope, _, _, p, _ = stats.linregress(x, min_ar[-num_no_improvement:])
        logging.debug(f"Found a loss slope of {slope}")
        if (slope < threshold and slope > -threshold) and p < 0.01:
            pass_check = False
    return pass_check

def sub_eq(model, value, eq_str):
    from brian2.equations.equations import parse_string_equations, SingleEquation
    new_equations = {}
    for eq in model._model['eqs'].values():
        if value == eq.varname:
            new_equations[eq.varname] = parse_string_equations(eq_str)[eq.varname]

        else:
            
            new_equations[eq.varname] = SingleEquation(eq.type, eq.varname,
                                                            dimensions=eq.dim,
                                                            var_type=eq.var_type,
                                                            expr=eq.expr,
                                                            flags=eq.flags)
    model._model['eqs'] = Equations(list(new_equations.values()))
    return model

