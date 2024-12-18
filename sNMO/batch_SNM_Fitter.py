''' Batch Single Neuron Model Fitting Script. Specficially focused for a
___________
Intended to be run from the command line with arguments. 
Input ARGS:
--inputFolder (path or str): Folder of NWB's to be fit with single neuron model
--outputFolder (path or str): Output folder of results
(optional)
--optimizer (str): one of 'ng' (nevergrad), 'skopt' (scikit-optimizer), or 'snpe' (deep learning)
--parallel (int): Number of threads to fit with (one cell per thread). [-1, 0, 1 will be not run in parallel]
'''
import argparse
import glob
import os
from multiprocessing import freeze_support
import logging
import pandas as pd
from brian2 import *
from joblib import Parallel, delayed
import json
import snm_fit as snm_fit
from loadNWB import *
import utils as ut
script_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
np.random.seed(46)
ut.utils.DEBUG = True

@ut.DEBUG_WRAPPER
def fit_cell(fp, optimizer, optimizer_settings, rounds=50, batch_size=500) -> pd.DataFrame:
    '''This is the primairy pass thru for cell fitting. It essentially takes a file path and optimizer keyword and tries to fit the cell
    _____
    takes:
    fp (str): a file path arguement pointing towards a single nwb
    optimizer (str): the string stating which optimizer to use'''

    cell_id = fp.split("\\")[-1].split(".")[0]
    temp_df = snm_fit.snmFitter().run_optimizer(fp, optimizer_settings, rounds=rounds, batch_size=batch_size, optimizer=optimizer)
    temp_df['id'] = [cell_id]
    return temp_df
    


def main(args, optimizer_settings):
    ''' This is the primairy function called upon callling the script from command line. This handles
    looking thru the folders for nwb files to fit. 
    ________
    takes:
    args (dict): these are the args passed in by the command line
    '''
    _path = [*glob.glob(args.inputFolder + '*.nwb'), *glob.glob(args.inputFolder + '*.abf')] #Glob allows unix like indexing. This tells it to look for all nwb files in the folder 
    prefit = glob.glob(args.outputFolder + '**//*.csv', recursive=True) #
    prefit_ids = [os.path.basename(x).split("_s")[0] for x in prefit]
    NWB_to_fit = _path #np.random.choice(_path, 150).tolist()
    np.random.seed()
    files_to_use = []
    for fp in NWB_to_fit: #here we are going to check if the file has already been fit, and if so, we will skip it
                cell_id = os.path.basename(fp).split('.')[0]
                if cell_id in prefit_ids:
                    continue
                else:
                    files_to_use.append(fp)
    file_id =[]
    full_df = pd.DataFrame()
    if args.parallel > 1:
        dataframes =  Parallel(n_jobs=args.parallel, backend='multiprocessing')(delayed(fit_cell)(fp, args.optimizer, optimizer_settings, args.rounds, args.batch_size) for fp in files_to_use)
    else:
        for fp in files_to_use:
                print(f"=== Opening {fp} ===")
                cell_id = os.path.basename(fp).split('.')[0]
                if cell_id in prefit_ids:
                    print("cell prev fit")
                    continue
                else:
                    res = fit_cell(fp, args.optimizer, optimizer_settings, args.rounds, args.batch_size)
                    full_df = pd.concat([full_df, res])
            
            
    full_df.to_csv(f'output//full_spike_fit.csv')


if __name__ == "__main__": ##If the script is called from the command line this runs
    freeze_support()
    parser = argparse.ArgumentParser(description='Batch Fit NWBs')

    _dir = os.path.dirname(__file__)
    parser.add_argument('--inputFolder', type=str, 
                        help='the input folder containing NWBs to be fit', default=(_dir + '//..//NWB_with_stim//macaque//pfc//'))
    parser.add_argument('--outputFolder', type=str,
                        help='the output folder for the generated data', default= _dir +'//out//')
    parser.add_argument('--optimizer', type=str, default='sbi',
                        help='the optimizer to use', required=False)
    parser.add_argument('--parallel', type=int, default=1,
                        help='number of threads to use (one cell per thread)', required=False)
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size of number of params to test in parallel per cell', required=False)
    parser.add_argument('--rounds', type=int, default=100,
                        help='number of rounds to optimize over', required=False)                        
    parser.add_argument('--optimizerSettings', type=str, default=script_path+"/optimizer_settings.json",
                        help='additional settings for the opitmizer to use', required=False)

    args = parser.parse_args()

    with open(args.optimizerSettings) as f:
        optimizer_settings = json.load(f)

    main(args, optimizer_settings)
