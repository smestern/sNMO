
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
from sbi import analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.base import infer
import json
from optimizer import snmOptimizer

with open('optimizer_settings.json') as f:
        optimizer_settings = json.load(f)

def generate_posterior(tag='', load_prev=True, ):
    '''
    This function is used to generate the posterior for use with SBI optimizer.
    Here we use the SBI optimizer to generate the posterior.
 
    '''
    print("== Loading Data ==")
    file_dir = os.path.dirname(os.path.realpath(__file__))
    default_dtype = torch.float32

    #%% Load and/or compute fixed params

    file_path = file_dir +'//..//NWB_with_stim//macaque//pfc//M02_MW_D4_C09.nwb'
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
    N = 10000
    batches=10
    opt = snmOptimizer(optimizer_settings['constraints'][optimizer_settings['model_choice']], N, batches, backend='sbi')
    
    def simulator_pass(x):
        #build dict
        x_dict = {}
        x = x.detach().numpy().T
        for i, key in enumerate(optimizer_settings['constraints'][optimizer_settings['model_choice']]):
                if key!='units':
                        x_dict[key] = x[i]
        param_dict = opt.attach_units(x_dict)
        model.set_params({'N': x.shape[1]})
        y = model.build_feature_curve(param_dict)
        return y


    model.set_params({'N': N})
    prior = opt.params
    
    theta_full = []
    res_full = []
    ##Now intialize the Neural Network. We tell it to run with a batch size same as the number of neurons we simulate in parallel
    ## from https://elifesciences.org/articles/56261j
    inference = SNPE(prior, density_estimator=utils.posterior_nn(model='maf', z_score_x=None), device="cpu") #
    #%% Run the inference or load previous results
    if load_prev:
        print("== Loading Previous Results ==")
        theta = torch.tensor(np.nan_to_num(np.load(f"{tag}_theta_ds.npy"), posinf=0.1, neginf=0.1, nan=0.1), dtype=default_dtype)
        res = torch.tensor(np.nan_to_num(np.load(f"{tag}_res_ds.npy"), posinf=0.12, neginf=0.1, nan=0.1), dtype=default_dtype)
    else:
        print("== Fitting Model ==")
        #use simulate for sbi to generate the data
        simulator, prior = prepare_for_sbi(simulator_pass, prior)
        theta, res = simulate_for_sbi(simulator, prior, num_simulations=int(N*batches), simulation_batch_size=N)
        np.save(f"{tag}_theta_ds.npy", theta.numpy())
        np.save(f"{tag}_res_ds.npy", res.numpy())

    #drop rows with only zeros
    theta = theta[~(res == 0).all(1)]
    res = res[~(res == 0).all(1)]
    # Now we need to run the inference for the posterior
    dens_est = inference.append_simulations(theta, res, proposal=prior).train(show_train_summary=True)
    posterior = inference.build_posterior(dens_est)
    posterior.set_default_x(res.numpy()[6])
    #try sampling the data?
    sample = posterior.sample((1000,))
    analysis.pairplot(sample)    
    with open(f"{tag}_post.pkl", "wb") as f:
            dump(posterior, f)
    with open(f"{tag}_dens_est.pkl", "wb") as f:
            dump(dens_est, f)
    with open(f"{tag}_prior.pkl", "wb") as f:
            dump(prior, f)
    plt.savefig(f"{tag}_posterior.png")
    plt.show()
    
    
if __name__=="__main__":
    generate_posterior('ADIF_posterior')



