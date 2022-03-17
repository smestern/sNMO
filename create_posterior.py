
import os
import argparse
from brian2 import *
from utils import *
from loadNWB import *
import pandas as pd
import numpy as np
from snm_fit import load_data_and_model
import torch
from pickle import dump, load
from sbi import utils as utils
from sbi import utils as sbutils
from sbi import analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.base import infer
import json
from optimizer import snmOptimizer

with open('optimizer_settings.json') as f:
        optimizer_settings = json.load(f)

def generate_posterior(tag=''):
    '''
    This function is used to generate the posterior for use with SBI optimizer.
    Here we use the SBI optimizer to generate the posterior.
 
    '''
    print("== Loading Data ==")
    file_dir = os.path.dirname(os.path.realpath(__file__))
    default_dtype = torch.float32

    #%% Load and/or compute fixed params

    file_path = file_dir +'//..//NWB_with_stim//macaque//V1//M19_JS_A1_C10.nwb'
    model = load_data_and_model(file_path, optimizer_settings)
    realX, realY, realC = model.realX, model.realY, model.realC
    idx_stim = np.argmin(np.abs(realX - 1))
    current_out = realC[:, idx_stim]
    #take only the current steps we want
    current_idx = [x in [-70, -50, 50, 70] for x in current_out]
    realX, realY, realC = realX[current_idx,:], realY[current_idx,:], realC[current_idx, :]
    #Compute Spike Times
    spike_time = detect_spike_times(realX, realY, realC, upper=1.15) 
    spiking_sweeps = np.nonzero([len(x) for x in spike_time])[0]
    rheobase = spiking_sweeps[0]
    non_spiking_sweeps = np.delete(np.arange(0, realX.shape[0]), spiking_sweeps)
    model.add_real_data(realX, realY, realC, spike_time, non_spiking_sweeps, spiking_sweeps)
    ##Global vars ###
    N = 15000
    batches=1
    opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']], N, batches, backend='sbi')
    
    model.set_params({'N': N})
    prior = opt.params

    theta_full = []
    res_full = []
    ##Now intialize the Neural Network. We tell it to run with a batch size same as the number of neurons we simulate in parallel
    ## from https://elifesciences.org/articles/56261j
    inference = SNPE(prior, device="cpu")
    #%% Run the inference
    print("== Fitting Model ==")
    for x in np.arange(batch/home/smestern/Desktop/NETSIM-dev-masteres):
        theta_temp = opt.ask(n_points=N)
        res_temp = model.build_feature_curve(theta_temp)
        theta_full.append(np.array(list(theta_temp.values())).T)#for whatever reason we need to transpose the rar
        res_full.append(res_temp)
    #Now run the inference for N of neuron simulations (1 batch). This runs the simulator function we provid with randomly selected params
    #and then computes the prob dist
    theta = torch.tensor(np.vstack(theta_full), dtype=default_dtype)
    res = torch.tensor(np.vstack(res_full), dtype=default_dtype)
    np.save(f"{tag}_theta_ds.npy", theta.numpy())
    np.save(f"{tag}_params_ds.npy", res.numpy())

    #%% Now we need to run the inference for the posterior
    dens_est = inference.append_simulations(theta, res, proposal=prior).train()
    posterior = inference.build_posterior(dens_est)
    with open(f"{tag}_post.pkl", "wb") as f:
            dump(posterior, f)

    with open(f"{tag}_prior.pkl", "wb") as f:
            dump(prior, f)
    
    
if __name__=="__main__":
    generate_posterior('ADIF_posterior')



