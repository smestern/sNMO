# -*- coding: utf-8 -*-
'''  '''
#%% Modified for fitting by GB and SM
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
from b2_model.optimizer import CustomPortfolio, snmOptimizer
with open('optimizer_settings.json') as f:
        optimizer_settings = json.load(f)

def generate_posterior(tag=''):

    print("== Loading Data ==")
    file_dir = os.path.dirname(os.path.realpath(__file__))
    default_dtype = torch.float32

    #%% Load and/or compute fixed params

    file_path = file_dir +'//..//NWB_with_stim//macaque//v1//M19_JS_A1_C10.nwb'
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
    batches=5
    opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']], N, batches, backend='sbi')
    
    model.set_params({'N': N})
    prior = opt.opt.params


    ##Now intialize the Neural Network. We tell it to run with a batch size same as the number of neurons we simulate in parallel
    ## from https://elifesciences.org/articles/56261j
    inference = SNPE(prior, device="cpu")
    #%% Run the inference
    print("== Fitting Model ==")
    for x in np.arange(batches):
        theta_temp = opt.ask(n_points=N)
        res_temp = model.build_feature_curve(theta_temp)
        if x == 0:
            theta = torch.tensor(opt.opt.param_list, dtype=default_dtype)
            res = torch.tensor(res_temp, dtype=default_dtype)
        else:
            theta = torch.vstack([theta, torch.tensor(opt.opt.param_list, dtype=default_dtype)])
            res = torch.vstack([res, torch.tensor(res_temp, dtype=default_dtype)])
    #Now run the inference for N of neuron simulations (1 batch). This runs the simulator function we provid with randomly selected params
    #and then computes the prob dist
    np.save(f"{tag}_theta_ds.npy", theta.numpy())
    np.save(f"{tag}_params_ds.npy", res.numpy())
    dens_est = inference.append_simulations(theta, res, proposal=prior).train()

    posterior = inference.build_posterior(dens_est)
    with open(f"{tag}_post.pkl", "wb") as f:
            dump(posterior, f)

    with open(f"_prior.pkl", "wb") as f:
            dump(prior, f)
    
    
if __name__=="__main__":
    generate_posterior()


# %%

if prefit_posterior is not None:
            with open(prefit_posterior, "rb") as f:
                pf = load(f, allow_pickle=True)
                self.posts.append(pf)
                self.proposal = pf
                self.prefit = True
            #with open(prefit_prior, "rb") as f:
                #pf = load(f, allow_pickle=True)
                #self.posts.append(pf)
                self.params = self.proposal
        else:
            self.proposal = self.params
            self.prefit = False

        if x_obs is not None:
            self.x_obs = x_obs
            self.params.set_default_x(x_obs)