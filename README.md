# sNMO : single neuron model optimzier (SNMO)

This project features several methods for fitting single neuron traces recorded during whole-cell patch clamp recordings. The primary focus is fitting square pulse sweeping stimualtion recordings. This projcet is focused/built for the neuronex-wm research group

## how-to
### basic fitting method

The project contains several methods for fitting single neurons, from high level methods to low level custom fitting methods.  

for neuronex-wm IRG2 members, there are two scripts to be used from the command line. These scripts are designed to fit single cells while handling most of the backend fitting procedures as well as data cleanup / QC.

#### -batch_SNM_fitter.py : A command line script used to iterate over a folder and fit the NWB's within. Allows parallelization, e.g. fitting one cell per thread. Call be called using the following commands
```
--inputFolder : a path to the folder containing the NWBS to be fitt
--outputFolder : a path to the output folder where fitting results will be saved, and progress will be reported
--optimizer : the optimizer to be used. Can be one of ['ng', 'skopt', 'sbi', 'snpe']
--parallel : number of cells to fit simultaneously. Numbers below 1 mean one cell fit per thread.
```

--snm_fit.py :

### adjusting optimzer settings

