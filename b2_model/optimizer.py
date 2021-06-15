import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO) #trying to log brian2 errors
logger = logging.getLogger(__name__)
import argparse
import os
from collections import OrderedDict
from pickle import dump, load

import nevergrad as ng
import nevergrad.common.typing as tp
import pandas as pd
import torch
from brian2 import *
from joblib import dump, load
from sbi import utils as sbutils
from sbi import analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.base import infer
from skopt import Optimizer, plots, space
from utils import *

from b2_model.adIF import adIFModel
from b2_model.error import weightedErrorMetric

default_dtype = torch.float32

TBPSAwithHam = ng.optimizers.Chaining([ng.optimizers.ScrHammersleySearch, ng.optimizers.NaiveTBPSA], ["num_workers"])
DEwithHam = ng.optimizers.Chaining([ng.optimizers.ScrHammersleySearch, ng.optimizers.TwoPointsDE], ["num_workers"])



def snmOptimizer(params_dict, batch_size, rounds, backend='ng', nevergrad_kwargs={}, skopt_kwargs={}, nevergrad_opt=DEwithHam, sbi_kwargs={}):
    ''' A backend agnostic optimizer, which should allow easier switching between skopt and nevergrad. internal optimizaion code handles returning the 
    params in a way that b2 models want  
    Takes:
    '''
    if backend == 'ng':
            return _internal_ng_opt(params_dict.copy(), batch_size, rounds, nevergrad_opt, nevergrad_kwargs=nevergrad_kwargs)  
    elif backend == 'skopt':
            return _internal_skopt(params_dict, batch_size, rounds)
    elif backend == 'sbi':
            return _internal_SBI_opt(params_dict.copy(), batch_size, rounds, **sbi_kwargs)
            
  

class _internal_ng_opt():
    def __init__(self, params_dict, batch_size, rounds, optimizer, nevergrad_kwargs={}):
        #Build Params
        self._units = [globals()[x] for x in params_dict.pop('units')]
        self._params = OrderedDict(params_dict)
        self._build_params_space()
        #intialize the optimizer
        self.rounds = rounds
        self.batch_size = batch_size
        self.optimizer = optimizer
        budget = (rounds * batch_size)
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
            
        param_dict = param_list.to_dict('list')
        for i, (key, val) in enumerate(param_dict.items()):
            param_dict[key] = val * self._units[i]
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
        
class _internal_skopt():
    def __init__(self, params, param_labels, batch_size, rounds, optimizer='RF', skopt_kwargs={}):
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

class _internal_Ax_opt():
    def __init__(self, params_dict, batch_size, rounds, x_obs=None, n_initial_sim=15000):
        self._units = [globals()[x] for x in params_dict.pop('units')]
        self._params = OrderedDict(params_dict)
        self._build_params_space()



class _internal_SBI_opt():
    ''''''
    def __init__(self, params_dict, batch_size, rounds, x_obs=None, n_initial_sim=500, prefit_posterior=None, prefit_prior=None):
        self._units = [globals()[x] for x in params_dict.pop('units')]
        self._params = OrderedDict(params_dict)
        self._build_params_space()
        #intialize the optimizer
        
        self.n_initial_sim = n_initial_sim
        
        self.rounds = rounds
        self.batch_size = batch_size
        self.optimizer = SNPE
        self.posts = []

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
            self.proposal.set_default_x(x_obs)
        budget = (rounds * batch_size)
        self.opt = self.optimizer(prior=self.params)
        self._bool_sim_run = False
        self.x_obs = x_obs

    def set_x(self, x):
        self.x_obs = x
        self.proposal.set_default_x(self.x_obs)

    def _build_params_space(self):
        ## Params should be a boxuniform
        lower_bound = []
        upper_bound = []
        for key, var in self._params.items():
            lower_bound.append(var[0])
            upper_bound.append(var[1])
        self.params = sbutils.BoxUniform(low=torch.tensor(lower_bound, dtype=default_dtype), high=torch.tensor(upper_bound, dtype=default_dtype)) #The SNPE-SBI wants the data in torch tensors which

    def ask(self, n_points=None):
        if n_points is None:
            n_points = self.batch_size
        if self._bool_sim_run == False:
            self.proposal.sample_with_mcmc = False
            self._bool_sim_run == True
        else:
            self.proposal.sample_with_mcmc = True
        self.param_list = self.proposal.sample((n_points,)).numpy()
        self.param_dict = {}
        for row, (key, value) in zip(self.param_list.T, self._params.items()):
            self.param_dict[key] = row
        for i, (key, val) in enumerate(self.param_dict.items()):
            self.param_dict[key] = val * self._units[i]
        return self.param_dict

    def tell(self, points, errors):
        assert errors.shape[0] == len(self.param_list)
        dens_est = self.opt.append_simulations(torch.tensor(self.param_list, dtype=default_dtype), torch.tensor(errors, dtype=default_dtype), proposal=self.proposal).train()
        posterior = self.opt.build_posterior(dens_est)
        self.posts.append(posterior)
        self.proposal = posterior.set_default_x(self.x_obs)
        return

    def get_result(self, points=50, from_cache=True):
        self.posts[-1].sample_with_mcmc = True
        if from_cache:
            posterior_samples = torch.tensor(self.param_list, dtype=default_dtype)
        else:
            posterior_samples = self.posts[-1].sample((points,), x=self.x_obs) #sample 500 points
        self.x_posterior_samples = posterior_samples
        log_prob = self.posts[-1].log_prob(posterior_samples, x=self.x_obs, norm_posterior=False).numpy()  # get the log prop of these points
        params = posterior_samples.numpy()[np.argmax(log_prob)] #Take the sample with the highest log prob
        res_dict = {}
        for i, (key, val) in enumerate(self._params.items()):
            res_dict[key] = params[i] * self._units[i]
        return res_dict

    def fit(self, model, id='default'):
        
        error_calc = weightedErrorMetric(weights=[1000, 1])
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
            analysis.plot.pairplot(self.x_posterior_samples, labels=[x for x in self._params.keys()])
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


def plot_feature_curve(x_o, model, res):
    model.set_params({"N": 1})
    x_best = model.build_feature_curve(res)
    plt.clf()
    plt.plot(x_o)
    plt.plot(np.ravel(x_best))

















#deprecated

class CustomPortfolio(ng.optimizers.Portfolio):
    """"""

    def __init__(self, parametrization, budget = None, num_workers = 1, optimizers = None) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None

        if optimizers  is None:
            optimizers = [
            ng.optimizers.CMA, 
            ng.optimizers.TwoPointsDE, 
            ng.optimizers.PSO,
            ng.optimizers.NaiveIsoEMNA,  
            ng.optimizers.ScrHammersleySearch,
            np.optimizers.SQP
        ]

        def intshare(n: int, m: int):
            x = [n // m] * m
            i = 0
            while sum(x) < n:
                x[i] += 1
                i += 1
            return tuple(x)

        #parallel opt check
        parallel_opt = [x.n]

        nws = intshare(num_workers, len(optimizers))
        self.which_optim = []
        for i, nw in enumerate(nws): #Using List comprehension did not work for whatever reason
            self.which_optim += [i] * nw

        assert len(self.which_optim) == num_workers
        self.optims: tp.List[base.Optimizer] = [opt(self.parametrization, num_workers=nw, budget=(budget // len(self.which_optim)) * nw) for opt, nw in zip(optimizers, nws)]
