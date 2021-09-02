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
from optimizer import snmOptimizer
import loadNWB as lnwb
from utils import *

warnings.filterwarnings("ignore")

#to allow parallel processing
prefs.codegen.target = 'cython'  # weave is not multiprocess-safe!
prefs.codegen.runtime.cython.multiprocess_safe = False
default_dtype = torch.float32

# === Global Settings ===
_rounds = 3
_batch_size = 15000
N = _batch_size
batches = _rounds

# === MAIN FUNCTION ===

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
    cell_id = file.split("\\")[-1].split(".")[0] #grab the cell id by cutting around the file path
    #adjust the constraints to the found CM or TAUM
    optimizer_settings['constraints'][optimizer_settings['model_choice']]['EL'] = [model.EL*1.01, model.EL*0.99]
    optimizer_settings['constraints'][optimizer_settings['model_choice']]['C'] = [model.C*0.75, model.C*1.25]
    optimizer_settings['constraints'][optimizer_settings['model_choice']]['taum'] = [model.taum*0.75, model.taum*1.25]
    print(f"== Loaded cell {cell_id} for fitting ==")
    if optimizer == 'skopt' or optimizer=='ng' or optimizer=='ax':
        results = biphase_opt(model, optimizer_settings, optimizer=optimizer, id=cell_id)
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
    
    global dt
    global spiking_sweeps
    global non_spiking_sweeps

    sweeps_to_use = None  
    file_path = file
    #set the stims
    lnwb.global_stim_names.stim_inc =  optimizer_settings['stim_names_inc']
    lnwb.global_stim_names.stim_exc = optimizer_settings['stim_names_exc']

    #load the data from the file
    cell_id = file.split("\\")[-1].split(".")[0] #grab the cell id by cutting around the file path
    realX, realY, realC,_ = lnwb.loadNWB(file_path, old=False)
    #crop the data
    index_3 = np.argmin(np.abs(realX[0,:]-2.50))
    ind_strt = np.argmin(np.abs((realX[0,:]-0.50)))

    if sweep_upper_cut != None:
        realX, realY, realC = realX[:,ind_strt:index_3], realY[:,ind_strt:index_3], realC[:,ind_strt:index_3]
    elif sweeps_to_use != None:
        #load -70, -50, 50 70
        idx_stim = np.argmin(np.abs(realX - 1))
        current_out = realC[:, idx_stim]
        #take only the current steps we want
        current_idx = [x in sweeps_to_use for x in current_out]
        if np.sum(current_idx) < len(sweeps_to_use):
            raise ValueError("The file did not have the requested sweeps")
        else:
            realX, realY, realC = realX[current_idx,ind_strt:index_3], realY[current_idx,ind_strt:index_3], realC[current_idx,ind_strt:index_3]
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
    #Compute cell params
    resistance = membrane_resistance_subt(realX[neg_current], realY[neg_current], realC[neg_current])
    taum = np.nanmean([exp_decay_factor(realX[x], realY[x], realC[x], plot=True) for x in non_spiking_sweeps])
    plt.title(f"{taum*1000} ms taum")
    plt.savefig(f"output//{cell_id}_taum_fit.png")
    Cm = (mem_cap((resistance*Gohm)/ohm, taum)) * farad
    Cm = Cm/pF
    taum *=1000

    #Create the model and attach the data. 
    model = brian2_model(model=optimizer_settings['model_choice'], param_dict={'EL': compute_el, 'dt':dt, '_run_time':2, 'C': Cm, 'taum': taum, 'id': cell_id})
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
    run_ng (bool): whether to run a few _rounds of the optimizer after sampling from SNPE (defaults to True)
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
    for x in  model.subthresholdSweep :
        temp = compute_steady_hyp(model.realY[x, :].reshape(1,-1), model.realC[x, :].reshape(1,-1))
        temp_min = compute_min_stim(model.realY[x, :], model.realX[x,:], strt=0.62, end=1.2)
        real_subt.append(temp)
        real_min.append(temp_min)
    real_subt = np.array(real_subt)        
    
    real_min = np.hstack(real_min)
    N = _batch_size
    np_o = np.hstack((real_fi, real_rmp, real_subt, real_min))
    x_o = torch.tensor(np.hstack((real_fi, real_rmp, real_subt, real_min)), dtype=torch.float32)
    
    

    #model.subthresholdSweep = None
    opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']], _batch_size, _rounds, backend='sbi', sbi_kwargs=dict(x_obs=x_o))
    #set the default X, seems to speed up sampling
    opt._rounds = 40
    opt.fit(model, id='test')
    # get top 100 samples
     ##TODO maybe take the mode of the highest samples for each column?
    #Take the 
    
    samples = opt.ask(n_points=500)
    log_prob = opt.posts[-1].log_prob(torch.tensor(opt.param_list), norm_posterior=False).numpy()
    
    top_100 = opt.param_list[np.argsort(log_prob[-10:])]
   
    #Now run a few _rounds of optimizer over the data
    high = np.apply_along_axis(np.amax,0, top_100) #take the high range of each column (each column is a different parameter)
    low = np.apply_along_axis(np.amin,0, top_100) #take the low range
    var_pairs = np.transpose(np.vstack((low, high))) #stack them into low->pairs for each row
    for i, (key, val) in enumerate(optimizer_settings['constraints'][optimizer_settings['model_choice']].items()):
        if key != 'units':
            optimizer_settings['constraints'][optimizer_settings['model_choice']][key] = var_pairs[i]


    if run_ng:
        results_out = _opt(model, optimizer_settings)  #returns a result containing the param - error matches
    elif run_ng_phase:
        results_out = biphase_opt(model, optimizer_settings)
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


    opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']].copy(), _batch_size, _rounds, backend=optimizer, nevergrad_opt=ng.optimizers.ParaPortfolio)#
    min_ar = []
    print(f"== Starting Optimizer with {_rounds} _rounds ===")
    for i in np.arange(_rounds):
        print(f"[CELL {id}] - iter {i} start")
        model.set_params({'N': _batch_size, 'refractory':0})
        t_start = time.time()
        param_list = opt.ask()
        param_dict = param_list
        print(f"sim {(time.time()-t_start)/60} min start")
        _, error_t, error_fi, error_isi, error_s = model.opt_full_mse(param_dict)
        error_fi = np.nan_to_num(error_fi, nan=999999) * 10
        error_t  = np.nan_to_num(error_t , nan=999999, posinf=99999, neginf=99999)
        y = error_t + error_fi + error_s
        #y = np.vstack([error_t, error_fi, error_s]).T
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
        print(f"[CELL {id}] - iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)} #with a min trace error {error_t[np.argmin(y)]} and FI error of {error_fi[np.argmin(y)]} and spike error of {error_s[np.argmin(y)]}") #    
    results = opt.get_result() #returns a result containing the param - error matches
    return results

def biphase_opt(model, optimizer_settings, optimizer='ng', id='nan'):
    """[summary]

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
    model.set_params({"VT":999999*mV})
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
    #model = sub_eq(model, 'a', f'a = (tauw/ms * {_})*nS : siemens')
    #create the cheap constraint
    def preserve_ratio(x):
        ratio = x['a'] / x['tauw'] 
        return (ratio <= high_ratio and ratio >= low_ratio)
    
    budget = int(_rounds * _batch_size) 


    param_ranges = cpy.deepcopy(optimizer_settings['constraints'][optimizer_settings['model_choice']])
    _, param_ranges['C'][0], param_ranges['C'][1] = mean_confidence_interval(top_c_val)
    _, param_ranges['taum'][0], param_ranges['taum'][1] = mean_confidence_interval(top_taum_val)

    opt = snmOptimizer(param_ranges, _batch_size, _rounds ** 20, backend=optimizer, nevergrad_opt=ng.optimizers.Portfolio)
    opt.opt.parametrization.register_cheap_constraint(preserve_ratio)
    min_ar = []
    for i in np.arange(_rounds):
        print(f"iter {i} start")
        model.set_params({'N': _batch_size, 'refractory':0})
        t_start = time.time()
        param_list = opt.ask()
        print(f"sim {(time.time()-t_start)/60} min start")
        _, error_t, error_fi, error_isi, error_s = model.opt_full_mse(param_list)
        y = error_fi + error_s + (error_t/100)
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
    """ Checks the minimum loss gradient for early stopping purposes. Using a simple linear regression of the minimum found in each local round.

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
        pass
    else:
        x = np.arange(0, num_no_improvement)
        slope, _, _, p, _ = stats.linregress(x, min_ar[-num_no_improvement:])
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