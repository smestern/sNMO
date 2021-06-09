from loadABF import *
from loadNWB import *
from utils import *
import os
import glob
import pandas as pd
from scipy import stats
from multiprocessing import freeze_support
from joblib import Parallel, delayed


parallel = False



def fit_cell(fp):
    cell_id = fp.split("\\")[-1].split(".")[0]
    realX, realY, realC = loadNWB(fp)
    if realX.shape[0] == 15:
            if np.any(cell_id==id_) and np.any(cell_id==id_qc):
                #try:
                    cell_params = mem_p.loc[cell_id].to_numpy()
                    spikes = detect_spike_times(realX, realY, realC)
                    resist = membrane_resistance(realX[0,:], realY[0,:], realC[0,:])
                    taum = exp_decay_factor(realX[0,:], realY[0,:], realC[0,:])
                    cm = mem_cap(resist, taum)
                    if len(spikes[12]) > 14:
                        
                        temp_df = snm_fit.run_optimizer(fp, cm, taum, rounds_=100, batch_size_=1000, optimizer='snpe')
                        temp_df['id'] = [cell_id]
                        temp_df['cm'] = [cm]
                        temp_df['taum'] = [taum]
                        return temp_df
    return pd.DataFrame

def main():
    _dir = os.path.dirname(__file__)
    _path = glob.glob(_dir +'//..//*.nwb')
    file_id =[]
    full_df = pd.DataFrame()
    for fp in _path:
            res = fit_cell(fp)
            full_df = full_df.append(res)
            
    full_df.to_csv(f'output//full_spike_fit.csv')


if __name__ == "__main__": 
    freeze_support()
    main()