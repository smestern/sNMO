
from loadNWB import *
from utils import *
import os
import glob
import pandas as pd
from scipy import stats
from multiprocessing import freeze_support
from joblib import Parallel, delayed
import numpy as np
from spike_train_utils import *
parallel = False



def fit_cell(fp):
    cell_id = fp.split("\\")[-1].split(".")[0]
    realX, realY, realC,_ = loadNWB(fp, old=False)
    idx_stim = np.argmin(np.abs(realX - 1))
    current_out = realC[:, idx_stim]
    return current_out

def all_in(ar2, ar1):
    ret = False
    in1d = np.in1d(ar1, ar2)
    if np.all(in1d):
        ret = True
    return ret


def main():
    _dir = os.path.dirname(__file__)
    _path = glob.glob(_dir + '//..//NWB_with_stim//macaque//v1//*.nwb')
    file_id =[]
    full_cells = []
    for fp in _path:
        try:
            current_out = fit_cell(fp)
            full_cells.append(current_out)
        except:
            pass
    np_val = equal_ar_size_from_list(full_cells, val=0)
    uni, count=np.unique(np_val, axis=0,return_counts=True)

    #Find single unique values
    np_val = equal_ar_size_from_list(full_cells, val=-9999)
    uni2, count2 = np.unique(np_val, return_counts=True)
    count2 = count2[uni2!=-9999]
    uni2 = uni2[uni2!=-9999]
    sort_count = np.sort(count2)[::-1]
    sort_uni = uni2[np.argsort(count2)[::-1]]
    sort_uni_gtr = sort_uni[:10]
    Row_contains = np.apply_along_axis(all_in, 1, np_val, sort_uni_gtr)
    Row_contains2 = np.apply_along_axis(all_in, 1, np_val, [-70,-50,50,70])
    print(f'found {np.sum(Row_contains2)} cells with common current')
    np.savetxt(_dir +"//output//common_current_in.csv", np.sort(sort_uni_gtr), delimiter=',', fmt='%.8f')


if __name__ == "__main__": 
    freeze_support()
    main()