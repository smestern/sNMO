# sNMO : single neuron model optimzier (SNMO) (and network fitting)

This project features several methods for fitting single neuron traces recorded during whole-cell patch clamp recordings. The primary focus is fitting square pulse sweeping stimualtion recordings. This project is focused/built for the neuronex-wm research group  

The actual package wraps several optimizers:  
-Nevergrad  
-SBI [https://www.mackelab.org/sbi/](https://www.mackelab.org/sbi/)  
-Ax  
-Scikit-optimize  

both sbi and nevergrad are recommended for models with wide / large parameter spaces.

## how-to
### basic fitting method (from command line)

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
   "stim_names_inc" : ["1000", "long"], // anything contained in these strings can be 
   "stim_names_exc" : ["rheo", "Rf50"],
   "sweeps_to_fit" : [],  //numeric sweeps to fit
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

Adding custom models is fairly easy. Custom models can be added currently by modifying models.py found in the b2_model subfolder. The model needs to be specified in the standard format required by brian2.
```
asqEx = dict(eqs=Equations('''
        dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - vc)/DeltaT) + I - w ) * (1./C) : volt (unless refractory)
        dw/dt = ( a*(v - EL) - w ) / tauw : amp (unless refractory)
        dvc/dt = (VT - vc) / tauVT : volt (unless refractory)
        Vcut = VT + (DeltaT * 5) : volt (constant over dt)
        gL = C/taum : siemens (constant over dt)
        tauw : second
        tauVT : second
        a : siemens
        b : amp
        C : farad
        taum : second
        EL : volt
        VT : volt
        VR : volt
        DeltaT : volt
        refrac : second
        bVT : volt
        I = in_current(t) : amp
        '''), threshold='v>Vcut', reset='v=VR; w+=b; vc+=bVT', refractory='refrac', method='euler', init_var=dict(vc='VT', v='EL')) 
```
eqs: specifies the ODE of the neuronal model
threshold: triggers the reset (keyword) code
refractory: is the refractory period after a spike where the model cannot fire.  
method: the method of integration. (optional)
init_var: a dictionary of intialized variables.


### Choosing an optimizer


