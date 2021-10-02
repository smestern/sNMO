# sNMO : single neuron model optimzier (SNMO) (and network fitting)

This project features several methods for fitting single neuron traces recorded during whole-cell patch clamp recordings. The primary focus is fitting square pulse sweeping stimualtion recordings. This projcet is focused/built for the neuronex-wm research group

## how-to
### basic fitting method

The project contains several methods for fitting single neurons, from high level methods to low level custom fitting methods. The goal of this project is to have functional programming interface, as well as an object oriented interface for better constraints.

for neuronex-wm IRG2 members, there are two scripts to be used from the command line. These scripts are designed to fit single cells while handling most of the backend fitting procedures as well as data cleanup / QC.

#### -batch_SNM_fitter.py : A command line script used to iterate over a folder and fit the NWB's within. Allows parallelization, e.g. fitting one cell per thread. Call be called using the following commands
```
--inputFolder : a path to the folder containing the NWBS to be fitt
--outputFolder : a path to the output folder where fitting results will be saved, and progress will be reported
--optimizer : the optimizer to be used. Can be one of ['ng', 'skopt', 'sbi', 'snpe', 'ax']
--parallel : number of cells to fit simultaneously. Numbers below 1 mean one cell fit per thread.
--optimizerSettings : points to a json file that contains additional settings (see below)
```

--snm_fit.py : //TODO DOCS


### adjusting optimizer settings
In addition to the main settings, additional optimizer settings such as variable constraints, model choice, protocols to be used can be adjusted by editings the optimizer_settings.json or creating your own optimizer settings json and passing it as an arg. At a minimum this file needs to specifiy model choice and constraints.
```
{  "model_choice" : "adEx", //the model to be fit, found in models.py
   "stim_names_inc" : ["1000", "long"],
   "stim_names_exc" : ["rheo", "Rf50"],
   "sweeps_to_fit" : [], 
   "constraints": { "adEx": { //the constraints for the variables found in the model.
                        "C" : [5, 350], //constraints should be [low, high] pairs
                        "taum": [9, 250],
                        "EL": [-90, -50],
                        "VT": [-70, -20],
                        "tauw" : [1, 500],
                        "a" : [-2, 5], 
                        "b" : [0.001, 100],
                        "VR" : [-80, -30],
                        "DeltaT": [0.5, 5],
                        "units" : ["pF", "ms", "mV", "mV", "ms", "nS", "pA", "mV", "mV"] //Units for the constraints, in order of how they are listed above
                        }                    
}
}

```

### adding custom models to fit

Adding custom models is fairly easy. Custom models can be added currently by modifying models.py found in the b2_model subfolder. The model needs to be specified in the standard
