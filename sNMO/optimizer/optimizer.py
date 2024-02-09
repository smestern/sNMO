import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO) #trying to log brian2 errors

import argparse
import os
from collections import OrderedDict
from pickle import load as pkl_load

import nevergrad as ng
import nevergrad.common.typing as tp
import pandas as pd
import torch
from brian2 import *
from joblib import dump, load
from sbi import utils as sbutils
from sbi import analysis
from sbi.inference import (SNPE,prepare_for_sbi)
from sbi.inference.base import infer
from skopt import Optimizer, plots, space
from joblib import Parallel, delayed

from ..utils import *
import time
try:
    from ax import *
    from ax.service.ax_client import AxClient
    from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
    from ax.modelbridge.registry import Models
    ax_loaded = True
except:
    logging.warning("Ax not installed, Ax optimizer unavailible")
    ax_loaded = False
default_dtype = torch.float32

TBPSAwithHam = ng.optimizers.Chaining([ng.optimizers.ScrHammersleySearch, ng.optimizers.NaiveTBPSA], ["num_workers"])
DEwithHam = ng.optimizers.Chaining([ng.optimizers.ScrHammersleySearch, ng.optimizers.TwoPointsDE], ["num_workers"])



def snmOptimizer(params_dict, batch_size, rounds, parallel_jobs=1, backend='ng', nevergrad_kwargs={}, skopt_kwargs={}, nevergrad_opt=DEwithHam, sbi_kwargs={}):
    ''' A backend agnostic optimizer, which should allow easier switching between skopt and nevergrad. internal optimizaion code handles returning the 
    params in a way that b2 models want  
    Takes:
    '''
    if backend == 'ng':
            return NG_optimizer(params_dict.copy(), batch_size, rounds, nevergrad_opt, nevergrad_kwargs=nevergrad_kwargs, parallel_jobs=parallel_jobs)  
    elif backend == 'skopt':
            return SKopt_optimizer(params_dict.copy(), batch_size, rounds, parallel_jobs=parallel_jobs)
    elif backend == 'sbi':
            return SBI_optimizer(params_dict.copy(), batch_size, rounds, parallel_jobs=parallel_jobs **sbi_kwargs)
    elif backend == 'ax':
            return Ax_optimizer(params_dict.copy(), batch_size, rounds, parallel_jobs=parallel_jobs)
            
class snMOptimizer():
    def __init__():
        raise "Not Implemented; this is an abstract class"
        pass
    def ask():
        pass
    def tell():
        pass
    def get_result():
        pass
    def optimize(self, func, error_func=None, kwargs={}):

        #this is a wrapper function to allow the user to pass in a function to optimize, and an error function to use if the user wants to use a different error function
        #than the one defined in the model. If no error function is passed, the func most return a single value which will be used as the error
        #or args that can be passed to the error function
        def _inner_round(param_dict):
            if error_func is not None:
                return error_func(func(param_dict))
            else:
                return func(param_dict)
        for i in np.arange(self.rounds): 
            param_dict = self.ask()
            errors = Parallel(n_jobs=self.parallel_jobs, backend='multiprocessing')(delayed(_inner_round)(param_dict) for i in np.arange(self.batch_size))
            self.tell(param_dict, errors)
        return self.get_result()
        
    def attach_units(self, param_dict):
        if isinstance(param_dict, list):
            new_list = []
            for row in param_dict:
                for i, (key, val) in enumerate(row.items()):
                    row[key] = val * self._units[i]
                new_list.append(row)
            param_dict = new_list
        elif isinstance(param_dict, dict):
            for i, (key, val) in enumerate(param_dict.items()):
                param_dict[key] = val * self._units[i]
        return param_dict

class NG_optimizer(snMOptimizer):
    def __init__(self, params_dict, batch_size, rounds, optimizer, nevergrad_kwargs={}, batch_trials=True, include_units=True, parallel_jobs=1):
        #Build Params
        self._units = [globals()[x] for x in params_dict.pop('units')]
        self._params = OrderedDict(params_dict)
        self._build_params_space()
        #intialize the optimizer
        self.rounds = rounds
        self.batch_size = batch_size
        self.optimizer = optimizer
        budget = (rounds * batch_size)
        self.batch_trials = batch_trials
        self.include_units = include_units
        self.opt = self.optimizer(parametrization=self.params, num_workers=batch_size, budget=budget, **nevergrad_kwargs)
    def _build_params_space(self):
        ## Params should be a dict
        var_dict = {}
        for key, var in self._params.items():
            temp_var = ng.p.Scalar(lower=var[0], upper=var[1], mutable_sigma=True) #Define them in the space that nevergrad wants
            var_dict[key] = temp_var
        self.params = ng.p.Dict(**var_dict)
    def ask(self, n_points=None):
        if n_points is None:
            n_points = self.batch_size
        self.points_list = []
        self.param_list =[]
        for p in np.arange(n_points):
                temp = self.opt.ask()
                self.param_list.append(temp.value)
                self.points_list.append(temp)
        param_list = pd.DataFrame(self.param_list)
        if self.batch_trials: 
            param_dict = param_list.to_dict('list')
        else:
            param_dict = param_list.to_dict('records')
        if self.include_units:
            param_dict = self.attach_units(param_dict)
        return param_dict
    def tell(self, points, errors):
        #assume its coming back in with the same number of points
        #otherwise this will break
        assert errors.shape[0] == len(self.points_list)
        for i, row in enumerate(self.points_list):
            self.opt.tell(row, errors[i])
    def get_result(self, with_units=True):
        best_val = self.opt.provide_recommendation().value
        if with_units:
            for i, (key, val) in enumerate(best_val.items()):
                best_val[key] = val * self._units[i]
        return best_val
        
class SKopt_optimizer(snMOptimizer):
    def __init__(self, params, param_labels, batch_size, rounds, optimizer='RF', parallel_jobs=1, skopt_kwargs={}):
        #Build Params
        self._params = params
        self._param_labels = param_labels
        self._build_params_space()
        self.rounds = rounds
        self.batch_size = batch_size
        #intialize the optimizer
        self.opt = Optimizer(dimensions=self.params, base_estimator=optimizer, n_initial_points=self.batch_size*3, acq_optimizer='sampling', n_jobs=-1)

    def _build_params_space(self):
        self.params = space.Space(self._params)
    
    def ask(self, n_points=None):
        if n_points is None:
            n_points = self.batch_size
        self.param_list = np.asarray(self.opt.ask(n_points=n_points)) ##asks the optimizer for new params to fit
        self.param_dict = {}
        for col, label in zip(self.param_list.T, self._param_labels):
            self.param_dict[label] = col
        return self.param_dict
    def tell(self, points, errors):
        #assume its coming back in with the same number of points
        #otherwise this will break
        assert errors.shape[0] == len(self.param_list)
        self.opt.tell(self.param_list.tolist(), errors.tolist())
    def get_result(self):
        results = self.opt.get_result() #returns a result containing the param - error matches
        out = results.x ##asks the optimizer for new params to fit
        param_dict = {}
        for label, value in zip(self._param_labels, out):
            param_dict[label] = value
        return param_dict

class SBI_optimizer(snMOptimizer):
    """ 
    """
    def __init__(self, params_dict, batch_size, rounds, x_obs=None, n_initial_sim=1000, prefit_posterior=None, sample_conditional=None, parallel_jobs=1):
        """ An SBI 'optimizer' which allows simple generation of a posterior, or multi-round inference to focus on a particular sample.

        Args:
            params_dict (dict): a dict containing the parameters to be optimized in high-low pairs
            batch_size (int): the batch size of the inference rounds
            rounds (int): number of inference rounds
            x_obs (ndarray, optional): the default x observation to be used for inference. Defaults to None.
            n_initial_sim (int, optional): number of initial points to sample from the prior. Defaults to 1500.
            prefit_posterior (str, optional): a file path pointing to a previously built prior distro. If None, prior will be built from scratch. Defaults to None.
            sample_conditional (bool, optional): whether to sample from the posterior or use the prior. Defaults to True.
        """
        self._units = [globals()[x] for x in params_dict.pop('units')]
        self._params = OrderedDict(params_dict)
        self._build_params_space()
        #intialize the optimizer
        
        self.n_initial_sim = n_initial_sim
        
        self.rounds = rounds
        self.batch_size = batch_size
        self.optimizer = SNPE
        self.posts = [] #will contain the generated posterior distributions. first item should be the prior

        if prefit_posterior is not None:
            logging.debug('Loading previous posterior from {}'.format(prefit_posterior))
            #if the user supplies a prefit posterior, load it
            with open(prefit_posterior, "rb") as f:
                pf = pkl_load(f)
                self.posts.append(pf)
                self.proposal = pf
                self.prefit = True
            #load the dense est path
            with open(prefit_posterior.replace('_post.pkl', '_dens_est.pkl'), "rb") as f:
                self.dense_est = pkl_load(f)
            #load the prior
            with open(prefit_posterior.replace('_post.pkl', '_prior.pkl'), "rb") as f:
                self.params = pkl_load(f)
            #load the data
            
            #self.data = np.load(prefit_posterior.replace('_post.pkl', '_params_ds.npy'))
            #load the theta
            #self.theta = np.load(prefit_posterior.replace('_post.pkl', '_theta_ds.npy'))
            
        else:
            self.proposal = self.params #our current proposal will be the prior
            self.posts.append(self.params) #add the prior to the list of posterior distributions
            self.prefit = False

        #set the initial x_obs
        if x_obs is not None:
            self.x_obs = x_obs
            try:
                self.proposal.set_default_x(x_obs)
                self.params.set_default_x(x_obs)
            except:
                logging.debug("x_obs not compatible with the proposal")

        #intialize the 'optimizer' in this case a SNPE
        budget = (rounds * batch_size)
        self.opt = self.optimizer(prior=self.params, density_estimator=sbutils.posterior_nn(model='maf', z_score_x=None))    
        self._bool_sim_run = False #flag to indicate whether or not the sim has been run this is to run many points on the intial prior
        
        if sample_conditional is not None and self.prefit:
            #if we have a prefit and the user wants to sample conditional, then we need to set the posterior to the prefit
            #self.proposal = self.opt.build_posterior(self.dense_est, sample_with='mcmc') #
            self.proposal.set_default_x(x_obs)
            #if the user wants to sample conditionally we need to use the new interface
            self.sample = self._sample_conditional
            self._conditional_constraints = sample_conditional #create the conditional funct etc
            
        else:
            self.sample = self._sample


    def set_x(self, x):
        """Sets the default X observation for the current proposal

        Args:
            x (ndarray): the x observation as a numpy array in the shape of (features,) or (1, features)
        """
        self.x_obs = x
        self.proposal.set_default_x(self.x_obs)

    def _build_params_space(self):
        """ Internal function to build params space in a way expected of SBI (box uniform)
        """
        ## Params should be a boxuniform
        lower_bound = []
        upper_bound = []
        for key, var in self._params.items():
            lower_bound.append(var[0])
            upper_bound.append(var[1])
        self.params = sbutils.BoxUniform(low=torch.tensor(lower_bound, dtype=default_dtype), high=torch.tensor(upper_bound, dtype=default_dtype)) #The SNPE-SBI wants the data in torch tensors which

    def ask(self, n_points=None):
        """ Ask for new points based on the current proposal (either the prior or a focused backend)

        Args:
            n_points (int, optional): Number of points to sample. If none, defaults to pre-provided batch_size. Defaults to None.

        Returns:
            points (dict): a dictonary of sampled points structured as {"param_name": numpy array of shape (points,)}
        """
        if n_points is None:
            n_points = self.batch_size
        if self._bool_sim_run == False:
            self.proposal.sample_with_mcmc = False
            self._bool_sim_run = True
        else:
            self.proposal.sample_with_mcmc = False
        self.param_list = self.sample((n_points,), self.proposal)
        self.param_dict = {}
        for row, (key, value) in zip(self.param_list.T, self._params.items()):
            self.param_dict[key] = row
        for i, (key, val) in enumerate(self.param_dict.items()):
            self.param_dict[key] = val * self._units[i]
        return self.param_dict

    def tell(self, points, errors):
        
        assert errors.shape[0] == len(self.param_list)
        dens_est = self.opt.append_simulations(torch.tensor(self.param_list, dtype=default_dtype), torch.tensor(errors, dtype=default_dtype), proposal=self.proposal).train(force_first_round_loss=True)
        
        #for multi round inference we need a restricted prior
        posterior = self.opt.build_posterior(dens_est).set_default_x(self.x_obs)
        restriction = sbutils.get_density_thresholder(posterior, quantile=1e-4)
        restricted_prior = sbutils.RestrictedPrior(self.proposal, restriction)
        self.posts.append(restricted_prior)
        self.proposal = restricted_prior
        return

    def get_result(self, points=50, from_cache=False, use_map=True):
        self.posts[-1].sample_with_mcmc = True
        #if the user asks, use MAP
        if use_map:
            self.posts[-1].sample_with_mcmc = True
            params = self.posts[-1].map().numpy()
        else:
            if from_cache:
                #if from cache just use the last sampled points
                posterior_samples = torch.tensor(self.param_list, dtype=default_dtype)
            else:
                posterior_samples = self.sample((points,), self.posts[-1]) #sample 500 points
            self.x_posterior_samples = posterior_samples
            log_prob = self.posts[-1].log_prob(posterior_samples.astype(np.float32)).numpy()  # get the log prop of these points
            params = posterior_samples[np.argmax(log_prob)] #Take the sample with the highest log prob
        res_dict = {}
        for i, (key, val) in enumerate(self._params.items()):
            res_dict[key] = params[i] * self._units[i]
        return res_dict

    def fit(self, model, id='default'):
        min_ar = []
        print(f"== Starting Optimizer with {self.rounds} rounds ===")
        for i in np.arange(self.rounds):
            print(f"[CELL {id}] - iter {i} start")
            model.set_params({'N': self.batch_size, 'refractory':0})
            t_start = time.time()
            if i == 0:
                #for the first round we ask for way more points
                param_list = self.ask(n_points=self.n_initial_sim)
                model.set_params({'N': self.n_initial_sim, 'refractory':0})
            else:
                param_list = self.ask()
            param_dict = param_list
            print(f"sim {(time.time()-t_start)/60} min start")
            y = model.build_feature_curve(param_dict)
            print(f"sim {(time.time()-t_start)/60} min end")
            self.tell(param_list, y) ##Tells the optimizer the param - error pairs so it can learn
            t_end = time.time()
            min_ar.append(np.sort(y)[:5])
            res = self.get_result(from_cache=False)
            #try:
            #analysis.plot.pairplot(self.x_posterior_samples, labels=[x for x in self._params.keys()])
            plt.savefig(f"output//{id}_{i}_pairplot.png")
            plot_trace(res, model)
                
            plt.savefig(f"output//{id}_{i}_fit_vm.png")
            plot_IF(res, model)
                
            plt.savefig(f"output//{id}_{i}_fit_IF.png")

            plot_feature_curve(self.x_obs, model, res)
            plt.savefig(f"output//{id}_{i}_feature_curve.png")
                #os.remove(f"output//{id}_{i-1}_fit_vm.png")
                #os.remove(f"output//{id}_{i-1}_fit_IF.png")
            #except:
               # pass
            #if len(min_ar) > 5:
                 #if _check_min_loss_gradient(min_ar, num_no_improvement=25, threshold=1e-5) == False:
                  #   break
            print(f"[CELL {id}] - iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)} ") #
        
    def fi_passthru(self, args):
            dict_in = {}
            for i, (row, (key, val)) in enumerate(zip(args.numpy().T, self._params.items())):
                dict_in[key] = row * self._units[i]
            self.model.set_params({'N': args.numpy().shape[0]})
            out = np.nan_to_num(np.hstack((self.model.build_FI_curve(dict_in))).reshape(args.numpy().shape[0], -1), posinf=0, neginf=0)
            return torch.tensor(out, dtype=default_dtype)

    def _sample(self, N, proposal):
        return proposal.sample(N).detach().numpy()
    
    def _sample_conditional(self, N, proposal):
        #proposal = 
        return self._construct_conditional_from_proposal(proposal, N)

    def _construct_conditional_from_proposal(self, proposal, N):
        #construct our fixed params
        fixed_params = []
        fixed_params_indicator = []
        for i, (key, val) in enumerate(self._params.items()):
            if key in self._conditional_constraints.keys():
                fixed_params.append(self._conditional_constraints[key])
                
            else:
                fixed_params.append(val[0])
                fixed_params_indicator.append(i)
        # put them in torch
        fixed_params = torch.tensor(np.array(fixed_params), dtype=default_dtype)
        fixed_params_indicator = torch.tensor(np.array(fixed_params_indicator), dtype=torch.int64)

        sample = proposal.sample_conditional(N, fixed_params, fixed_params_indicator, ).detach().numpy()
        #now we need to put the fixed params back in
        conditions_ =[]
        for i, (key, val) in enumerate(self._params.items()):
            if key in self._conditional_constraints.keys():
                conditions_.append(np.full(N, self._conditional_constraints[key]))
        conditions_ = np.vstack(conditions_)
        sample = np.hstack((conditions_.T, sample))
        return sample

if ax_loaded:
    class Ax_optimizer():
        class dummy_metric(Metric):
            def fetch_trial_data(self, trial):  
                records = []
                for i, (arm_name, arm) in enumerate(trial.arms_by_name.items()):
                    params = arm.parameters
                    records.append({
                        "arm_name": arm_name,
                        "metric_name": self.name,
                        "mean": trial.run_metadata['snm_error'][i],
                        "sem": 0.0,
                        "trial_index": trial.index,
                    })
                return Data(df=pd.DataFrame.from_records(records))

        class dummy_runner(Runner):
                def __init__(self, parent_obj):
                    self.parent_obj = parent_obj
                def run(self, trial):
                    return {"name": str(trial.index), "snm_error": self.parent_obj._error}

        def __init__(self, params_dict, batch_size, rounds, device_gp='cuda', num_sobol=3,parallel_jobs=1):
            
            self._units = [globals()[x] for x in params_dict.pop('units')]
            self._params = OrderedDict(params_dict)
            self._build_params_space()
            self.rounds = rounds
            self.batch_size = batch_size
            if device_gp=='cuda':
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cpu")
            self.num_sobol = num_sobol
            self.gs = self._build_gs()
            self.opt = AxClient(enforce_sequential_optimization=False, generation_strategy=self.gs)
            self.exp = Experiment(name='snm_fitting', search_space=SearchSpace(parameters=self.params))
            #self.opt.create_experiment(parameters=self.params, choose_generation_strategy_kwargs={"max_parallelism_override": batch_size}, objective_name='snm_fit')
            self.metric = Ax_optimizer.dummy_metric(name="booth")
            optimization_config = OptimizationConfig(
                    objective = Objective(
                        metric=self.metric, 
                        minimize=True,
                    ),
                    )
            self.exp.runner = Ax_optimizer.dummy_runner(parent_obj=self)
            self.exp.optimization_config = optimization_config
            self.model = Models.SOBOL
            self._internal_run_count = int(-1)

        def _build_params_space(self):
            self.params = []
            for key, val in self._params.items():
                self.params.append(RangeParameter(name=key, lower=float(val[0]), upper=float(val[1]), parameter_type=ParameterType.FLOAT))

        def _build_gs(self):
            
            gs = GenerationStrategy(
                steps=[
                    # I omit a few settings when copying from the auto-selected generation strategy above, since those
                    # settings are just the defaults, but you can include them too if you'd like. Refer to the docs on
                    # `GenerationStep` for meanings of these settings: 
                    # https://ax.dev/api/modelbridge.html#ax.modelbridge.generation_strategy.GenerationStep
                    GenerationStep(
                        model=Models.SOBOL, 
                        num_trials=self.batch_size*self.num_sobol, 
                        min_trials_observed=self.batch_size*self.num_sobol, 
                        model_kwargs={'deduplicate': True, 'seed': None}, 
                        # Any kwargs you want passed to `modelbridge.gen`
                    ),
                    GenerationStep(
                        model=Models.GPEI,
                        num_trials=-1,
                        max_parallelism=self.batch_size,  # Can set higher parallelism if needed
                        model_kwargs = {"torch_dtype": torch.float, "torch_device": self.device},
                    )
                ]
            )
            return gs
        
        def ask(self, n_points=None):
            if n_points is None:
                n_points = self.batch_size
            self._internal_run_count +=1
            self.points_list = []
            self.param_list =[]
            self.points_list = self.gs.gen(experiment=self.exp, n=n_points)
            for a in self.points_list.arms:
                self.param_list.append(a.parameters)
            param_list = pd.DataFrame(self.param_list)
                
            param_dict = param_list.to_dict('list')
            for i, (key, val) in enumerate(param_dict.items()):
                param_dict[key] = val * self._units[i]
            self.exp.new_batch_trial(generator_run=self.points_list)
            return param_dict
    
        def tell(self, points, errors):
            #assume its coming back in with the same number of points
            #otherwise this will break
            #assert errors.shape[0] == len(self.points_list)
            self._error = errors
            self.exp.trials[self._internal_run_count].run().mark_completed()
            data = self.exp.fetch_data()
            
        def get_result(self, with_units=True):
            best_parameters = self.find_best_parameters()
            best_val={}
            if with_units:
                for i, (key, val) in enumerate(best_parameters.items()):
                    best_val[key] = val * self._units[i]
            return best_val

        def find_best_parameters(self):
            data = self.exp.fetch_data().df
            # get the error
            error = data['mean'].to_numpy()
            min_idx = np.argmin(error)
            #get the trial and arm name of min
            arm_name = data['arm_name'].to_numpy()[min_idx]
            trial_index = data['trial_index'].to_numpy()[min_idx]
            #get those params
            param_dict = self.exp.arms_by_name[arm_name].parameters
            return param_dict


def plot_feature_curve(x_o, model, res):
    model.set_params({"N": 1})
    x_best = model.build_feature_curve(res)
    plt.clf()
    plt.plot(x_o)
    plt.plot(np.ravel(x_best))














