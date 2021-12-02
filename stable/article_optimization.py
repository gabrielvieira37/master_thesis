import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from collections import defaultdict
from scipy.optimize import minimize
import datetime
from dateutil.relativedelta import relativedelta
import time as tm
import os
import logging
from matplotlib import cm
import torch
import torch.nn as nn
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch

torch.random.manual_seed(123)
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'

from utils import (
    data_split, create_w_benchmark, plot_splitted_data,
    utility_function, constrain_weights, compute_transaction_costs,
    ParametricPortifolioNN, loss_fn, convert_to_nn_variables, weight_reset,
    strict_decreasing, strict_increasing
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
format_string = '%(asctime)s [%(process)d.%(thread)d] %(levelname)-3s %(name)-3s  %(funcName)s(%(lineno)d): %(message)s'
formatter = logging.Formatter(format_string)
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.tune.suggest.bayesopt").setLevel(logging.ERROR)

class ParametricPortifolio():
    """
    Parametric portifolio using firm characteristics 
    to adjust portifolio weights.
    """

    def __init__(self, data_path, risk_constant, train_split, val_split, learning_rate, l2_regularization, epochs_size, patience, plot_weights):
        """
        Initialize object with data path, risk constant, transaction cost and
        the percentage of train split.

        Parameters
        ----------

        """
        self.data_path = data_path
        self.risk_constant = risk_constant
        self.train_split = train_split
        self.val_split = val_split
        self.patience = patience
        self.plot_weights = plot_weights

        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.epochs_size = epochs_size

        self.training = True
        self.mean_mean = {}
        self.mean_std = {}

        # 01/01/1995
        base = datetime.datetime.fromtimestamp(788925600)
        # 203 months
        self.timestamp = pd.date_range(base, base + relativedelta(months=+203), freq='MS').strftime("%Y-%b").tolist()

        # Train values
        self.mean_obj_r_runs = []
        self.mean_obj_r_val_runs = []
        self.mean_r_runs = []
        self.mean_constrained_r_runs = []
        self.mean_constrained_transaction_r_runs = []
        self.benchmark_mean_return_runs = []

        # Train values nn
        self.nn_loss_runs = []
        self.nn_return_runs = []
        self.nn_return_runs_std = []

        self.nn_val_loss_runs = []
        self.nn_val_return_runs = []
        self.nn_val_return_runs_std = []

        # Test values
        self.benchmark_test_r_runs = []
        self.benchmark_test_r_runs_std = []

        self.test_r_runs = []
        self.test_r_runs_std = []

        self.test_r_constrained_runs =[]
        self.test_r_constrained_runs_std=[]

        self.test_r_constrained_transaction_runs=[]
        self.test_r_constrained_transaction_runs_std=[]

        # Test values NN
        self.test_r_nn_runs = []
        self.test_r_nn_runs_std = []

        self.test_r_nn_constrained_runs = []
        self.test_r_nn_constrained_runs_std =[]


        self.test_r_nn_constrained_transaction_runs = []
        self.test_r_nn_constrained_transaction_runs_std =[]

        self.weights_computed = {'optimized':[], 'optimized_constrained':[], 'nn_optimized':[], 'nn_optimized_constrained':[]}

        self.results_comparison = {}
        comparison_fields = ["cdi", "ibov"]
        calculated_fields = ["nn", "nn_constraint", "opt", "opt_constraint"]

        for comp_field in comparison_fields:
            for cal_field in calculated_fields:
                self.results_comparison[f"{cal_field}_{comp_field}_comparison"] = {}

    # TO-DO: Change it to load any number of characteristics
    def load_data(self):
        """
        Load returns and firms characteristics from path.

        Parameters
        ----------
        data_path: str
            The path where data are.

        Returns
        -------
        lreturn: pandas.DataFrame
            Lagged return of the firms.

        mcap: pandas.DataFrame
            Market capitalization of the firms

        book_to_mkt_ratio: pandas.DataFrame
            Book to market ratio of the firms

        monthly_return: pandas.DataFrame
            Monthly return of the firms.

        """
        self.lreturn = pd.read_csv(os.path.join(self.data_path, "monthly_lagged_return.csv"))
        self.mcap = pd.read_csv(os.path.join(self.data_path, "monthly_market_cap.csv"))
        self.book_to_mkt_ratio = pd.read_csv(os.path.join(self.data_path, "monthly_book_to_mkt_ratio.csv"))
        self.monthly_return = pd.read_csv(os.path.join(self.data_path, "monthly_return.csv"))
        self.lreturn.fillna(method='bfill', inplace=True)
        self.lreturn.fillna(method='ffill', inplace=True)
        self.mcap.fillna(method='bfill', inplace=True)
        self.mcap.fillna(method='ffill', inplace=True)
        self.book_to_mkt_ratio.fillna(method='bfill', inplace=True)
        self.book_to_mkt_ratio.fillna(method='ffill', inplace=True)
        self.monthly_return.fillna(method='bfill', inplace=True)
        self.monthly_return.fillna(method='ffill', inplace=True)

        self.market_cost = compute_transaction_costs(self.data_path)
        self.cdi_return = pd.read_csv(os.path.join(self.data_path, "cdi_return.csv"))
        self.ibov_return = pd.read_csv(os.path.join(self.data_path, "ibov_return.csv"))

        LOGGER.info("Loaded data successfully")
    
    # TO-DO: Change it to create any number of characteristics
    def create_characteristics(self, me_df, mom_df, btm_df, return_df):
        """
        Create firm characteristics complete dataframe

        Parameters
        ----------
        me_df: pandas.DataFrame
            Market capitalization datafame.

        mom_df: pandas.DataFrame
            Lagged return dataframe.

        btm_df: pandas.DataFrame
            Book to market ratio dataframe.

        stock_names: [str,]
            List of stock names as strings.


        Returns
        -------
        firm_characteristics: pandas.DataFrame
            Normalized firm characteristcs dataframe.
        r: numpy.array
            Return of the stocks through time.
        time: int
            Number of time periods this slice of firm characterists are evaluating.
        number_of_stocks: int
            Number of stocks that we created the firm characteristics dataframe.


        """
        stocks_names = self.stocks_names

        firm_characteristics = defaultdict(list)
        time = return_df.shape[0] 
        number_of_stocks = len(stocks_names)
        r = np.empty(shape=(number_of_stocks, time))
        
        characteristics_names = ["me", "btm", "mom"]
        for i, name in enumerate(stocks_names):
            me = me_df.get(name)
            mom = mom_df.get(name)
            btm = btm_df.get(name)
            mr = return_df.get(name) 

            firm_characteristics[(i,'me')] = me.fillna(method='bfill')
            firm_characteristics[(i,'btm')] = btm.fillna(method='bfill')
            firm_characteristics[(i,'mom')]= mom.fillna(method='bfill')

            r[i] = mr.fillna(method='bfill')

        LOGGER.info("Created firm characteristics dataframe")
        firm_characteristics = pd.DataFrame(firm_characteristics)
        firm_characteristics = self.normalize_characteristics(firm_characteristics, characteristics_names)

        return firm_characteristics, r, time, number_of_stocks
    

    def normalize_characteristics(self, firm_characteristics, characteristics_names):
        """
        Normalize firm characteristics by subtracting it by the mean of all stocks at each time,
        and dividing it by the std of all stocks at each time. Making it have unit variance.

        Parameters
        ----------
        firm_characteristics: pandas.DataFrame
            Unnormalized firm characteristcs dataframe.
        
        characteristics_names: [str, ]
            List of characteristics names. e.g.: 'me','btm'.

        Returns
        -------
        firm_characteristics: pandas.DataFrame
            Normalized firm characteristcs dataframe.
        """


        # Normalization using mean and std of each feature to make its sum going to zero each time step.
        # Normalize within cross-section
        LOGGER.info(f"IsTraining: {self.training}")
        for name in characteristics_names:
            #Normalize firm characteristics for all stocks
            # if self.training == True:
            mean_mean = firm_characteristics.T.loc[(slice(None), name), :].mean()
            mean_std = firm_characteristics.T.loc[(slice(None), name), :].std()
            #     self.mean_mean[name] = mean_mean
            #     self.mean_std[name] = mean_std

            # mean_mean = self.mean_mean[name]
            # mean_std = self.mean_std[name]

            firm_characteristics.T.loc[(slice(None), name), :] -= mean_mean
            firm_characteristics.T.loc[(slice(None), name), :] /= mean_std
        
        # import pdb; pdb.set_trace()
        LOGGER.info("Normalized firm characteristics")
        return firm_characteristics
    

    def optimizing_step(self, firm_characteristics, r, time, number_of_stocks, theta0, w_benchmark, firm_characteristics_val, r_val, time_val, w_benchmark_val):
        """
        Find the coefficient theta that best maps the firm characteristics and the returns.

        Parameters
        ----------
        firm_characteristics: pandas.DataFrame
            Normalized firm characteristcs dataframe.
        r: numpy.array
            Return of the stocks through time.
        time: int
            Number of time periods this slice of firm characterists are evaluating.
        number_of_stocks: int
            Number of stocks that we created the firm characteristics dataframe.
        theta0: numpy.array
            Initial coefficients mapping the return and the firm characteristics.
        risk_constant: int
            Risk constant, increase it to become more risk averse.
        w_benchmark: numpy.array
            Benchmark weights, used to create optimized weights.
        transaction_cost: float
            Define transaction cost.
        firm_characteristics_val: pandas.DataFrame

        r_val: numpy.array

        time_val: int

        w_benchmark_val: numpy.array

        Returns
        -------
        self.sol: scipy.optimize.OptimizeResult
            Solution object used to retrieve theta optimized and optimization information.
        self.mean_obj_r: [float, ]
            List of objective values through each optimization step.
        self.mean_r: [float, ]
            List of return using optimized weights through each optimization step.
        self.mean_obj_r_val: [float, ]
            List of objective validation values through each optimization step.
        self.mean_r_val: [float, ]
            List of validation return using optimized weights through each optimization step.
        self.mean_constrained_r: [float, ]
            List of return constrained using weights with 30% of leverage.
        self.mean_constrained_transaction_r: [float, ]
            List of return constrained adding transaction costs.
        """
        LOGGER.info("Started optimization step.")

        # Get constants
        risk_constant = self.risk_constant
        def objective(theta):
            """
            Optimize the returns of the utility function through time.

            Parameters
            ----------
            theta: numpy.array
                Coefficients mapping the return and the firm characteristics.
            
            Returns
            -------
            value: float
                Average value of return of utility function through time.
            """
            w = np.empty(shape=(number_of_stocks, time))
            for i in range(number_of_stocks):
                w[i] = w_benchmark[i].copy() + (1/number_of_stocks)*theta.dot(firm_characteristics[i].copy().T)
            return -np.mean(np.sum(utility_function(risk_constant, w[:,:-1]*r[:,1:]), axis=0))
        
        mean_obj_r = []
        mean_obj_r_val = []
        mean_r = []
        mean_r_val = []
        mean_constrained_r = []
        mean_constrained_transaction_r = []
        
        def callback_steps(thetaI):
            """
            Callback of optimization function.

            Parameters
            ----------
            thetaI: numpy.array
                Optimized theta vector through the ith optmization step.

            """        
            LOGGER.debug(f"i:{len(mean_obj_r)}, theta i: {thetaI}, f(theta):{objective(thetaI)}")
            mean_obj_r.append(-objective(thetaI))

            w_iter = np.empty(shape=(number_of_stocks, time))
            w_iter_val = np.empty(shape=(number_of_stocks, time_val))
            
            for i in range(number_of_stocks):
                w_iter[i] = w_benchmark[i] + (1/number_of_stocks)*thetaI.dot(firm_characteristics[i].copy().T)
                w_iter_val[i] = w_benchmark_val[i] + (1/number_of_stocks)*thetaI.dot(firm_characteristics_val[i].copy().T)
            w_iter_constrained = constrain_weights(w_iter.copy())

            obj_val = np.mean(np.sum(utility_function(risk_constant, w_iter_val[:,:-1]*r_val[:,1:]), axis=0))
            mean_obj_r_val.append(obj_val)

            mean_r_val.append(np.mean(np.sum(w_iter_val[:,:-1]*r_val[:,1:], axis=0)))   
            mean_r.append(np.mean(np.sum(w_iter[:,:-1]*r[:,1:], axis=0)))
            mean_constrained_r.append(np.mean(np.sum(w_iter_constrained[:,:-1]*r[:,1:] , axis=0 )))
            mean_constrained_transaction_r.append(np.mean(np.sum(w_iter_constrained[:,:-1]*r[:,1:]*self.train_market_cost[:-1].T, axis=0)) )

        sol = minimize(objective, theta0, callback=callback_steps, method='BFGS')
        self.sol = sol
        self.mean_obj_r  = mean_obj_r
        self.mean_obj_r_val = mean_obj_r_val
        self.mean_r = mean_r
        self.mean_r_val = mean_r_val
        self.mean_constrained_r = mean_constrained_r
        self.mean_constrained_transaction_r = mean_constrained_transaction_r
        LOGGER.info("Finished optimization step.")

    def nn_hyperparameter_optimizer(self, config):
        """
        Optimize hyperparameters using baysean optimization

        Parameters
        ----------
        self.nn_optimizer: function
            Function to perform portifolio optimization using neural networks.
        self.torch_characteristics: torch.Tensor

        self.torch_r: torch.Tensor

        self.torch_benchmark: torch.Tensor

        self.number_of_stocks: Int

        self.torch_characteristics_val: torch.Tensor

        self.torch_r_val: torch.Tensor

        self.torch_benchmark_val: torch.Tensor
            
        config: Dict
            Dictionary containing all hyperparameters to be used.
        
        Return
        ------
            Does not return anything but report to hyperparameter 
            optimization the result using current hyperparameter configuration.
        """

        
        torch_characteristics=self.torch_characteristics
        torch_r=self.torch_r
        torch_benchmark=self.torch_benchmark
        number_of_stocks=self.number_of_stocks
        torch_characteristics_val = self.torch_characteristics_val
        torch_r_val = self.torch_r_val
        torch_benchmark_val = self.torch_benchmark_val

        optimized_nn = self.nn_optimizer(
            torch_characteristics, torch_r, torch_benchmark, number_of_stocks, 
            torch_characteristics_val, torch_r_val, torch_benchmark_val, config
            )
        
        # Last epoch results
        mean_loss = self.nn_val_loss[-1]
        mean_return =  self.nn_val_return[-1]
        mean_std = self.nn_val_return_std[-1]

        tune.report(
            mean_return=mean_return,
            mean_sharpe_ratio=mean_return/mean_std,
            mean_loss=mean_loss
        )

    def get_best_hyperparameter_config(self):
        """
        Get best hyperparameter config using hyperparamter optimization
        
        Parameters
        ----------
        self.nn_hyperparameter_optimizer: function
            Function to map hyperparamter optimization with nn optimization 

        Returns
        -------
        best_config: dict
            Dictionary containing best hyperparameter config such as : 
            learning_rate, l2_regularization, epochs_size, adam_betha1, adam_betha2 and patience.

        """

        LOGGER.info("Started NN hyperparameter optimization.")
        
        config = {        
            'learning_rate':tune.uniform(0.01, 0.1),
            'l2_regularization':tune.uniform(1e-7, 1e-2),
            'epochs_size':tune.uniform(500, 3000),
            'adam_betha1':tune.uniform(0.85, 0.95),
            'adam_betha2':tune.uniform(0.99, 0.999),
            'patience':tune.uniform(2, 10),
        }

        reporter = tune.CLIReporter(max_progress_rows=10)
        reporter.add_metric_column("mean_return")
        reporter.add_metric_column("mean_sharpe_ratio")
        bayesopt = BayesOptSearch(verbose=0)
        analysis = tune.run(
            self.nn_hyperparameter_optimizer, config=config, progress_reporter=reporter, search_alg=bayesopt,
            num_samples=100, metric="mean_loss", mode="min", verbose=0
        )

        LOGGER.info("Finished NN hyperparameter optimization.")
        best_config =  analysis.get_best_config(metric='mean_loss', mode='min')
        LOGGER.info(f"Best NN config: {best_config}")
        return best_config

    def nn_optimizer(self, torch_characteristics, torch_r, torch_benchmark, number_of_stocks, torch_characteristics_val, torch_r_val, torch_benchmark_val, config):
        """
        Optimize and fing theta using a neural network. Theta is the weights of the optimized NN.

        Parameters
        ----------
        torch_characteristics:
        torch_r, torch_benchmark:
        number_of_stocks:
        torch_characteristics_val:
        torch_r_val:
        torch_benchmark_val:
        config:
        
        Returns
        -------
        self.nn_loss:
        self.nn_return:
        self.nn_return_std:
        self.nn_val_loss:
        self.nn_val_return:
        self.nn_val_return_std:
        optimized_nn:

        """
        LOGGER.info("Start neural network optimization.")
        portifolio = ParametricPortifolioNN(torch_benchmark[:-1], torch_r[1:], self.risk_constant, number_of_stocks)
        portifolio_val = ParametricPortifolioNN(torch_benchmark_val[:-1], torch_r_val[1:], self.risk_constant, number_of_stocks)
        portifolio.apply(weight_reset)

        learning_rate = config['learning_rate']
        l2_regularization = config['l2_regularization']
        epochs_size = config['epochs_size']
        epochs_size = int(epochs_size)

        adam_betha1 = config['adam_betha1']
        adam_betha2 = config['adam_betha2']
        patience = config['patience']
        patience = int(patience)

        torch_r_val = torch_r_val[1:]
        
        opt = torch.optim.Adam(portifolio.parameters(), betas=(adam_betha1, adam_betha2) , lr=learning_rate, weight_decay=l2_regularization)

        patience_counter = 0
        min_loss_val = 0
        loss_values = []
        return_values_mean = []
        return_values_std = []

        loss_values_val =[]
        return_values_mean_val = []
        return_values_std_val = []
        for i in range(epochs_size):
            opt.zero_grad()
            value, r_ = portifolio(torch_characteristics[:-1])
            loss = loss_fn(value)
            loss_values.append(loss.item())
            r_p = torch.sum(r_,-1)
            mean_r_p = torch.mean(r_p).detach().numpy()
            std_r_p = torch.std(r_p).detach().numpy()

            portifolio_val.weights = portifolio.weights
            value_val, r_val = portifolio_val(torch_characteristics_val[:-1])
            loss_val = loss_fn(value_val)
            r_p_val = torch.sum(r_val,-1)
            mean_r_p_val = torch.mean(r_p_val).detach().numpy()
            std_r_p_val = torch.std(r_p_val).detach().numpy()

            return_values_mean_val.append(mean_r_p_val)
            return_values_std_val.append(std_r_p_val)
            loss_values_val.append(loss_val.item())
                
            
            if i%100==0:
                theta = portifolio.weights.state_dict()['weight'].detach().numpy()
                LOGGER.debug(f"i:{i}, theta i: {theta}, f(theta):{loss.item()},  Return_mean:{mean_r_p}, Return_std{std_r_p}")

            return_values_mean.append(mean_r_p)
            return_values_std.append(std_r_p)
            loss.backward()
            opt.step()

            # New Early stopping
            if len(loss_values_val)==1:
                min_loss_val = loss_values_val[-1]
            else:
                if min_loss_val > loss_values_val[-1]:
                    min_loss_val = loss_values_val[-1]
                    patience_counter=0
                else:
                    patience_counter+=1

            if patience_counter == patience:
                break
            
            # If the difference between validation loss and training loss is greater than 80% break
            relative_diff_loss =  (loss_values[-1] - loss_values_val[-1])/loss_values_val[-1]
            if i%50==0:
                LOGGER.debug(f"Relative difference of validation and training loss: {relative_diff_loss}")
            if abs(relative_diff_loss) > 0.8:
                break

            # ## Early stopping
            # if len(loss_values) > patience:
                
            #     comparison_loss = loss_values[-patience:]
            #     comparison_loss_val = loss_values_val[-patience:]

            #     logic_loss = strict_decreasing(comparison_loss)
            #     logic_loss_val = strict_increasing(comparison_loss_val)

            #     if logic_loss==True and logic_loss_val==True:
            #         break
        
        theta = portifolio.weights.state_dict()['weight'].detach().numpy()
        self.nn_loss = loss_values
        self.nn_return = return_values_mean
        self.nn_return_std = return_values_std
    
        self.nn_val_loss = loss_values_val
        self.nn_val_return = return_values_mean_val
        self.nn_val_return_std = return_values_std_val
    

        optimized_nn = portifolio
        return optimized_nn

    # TO-DO: Change it to load any number of characteristics
    # BUY-HOLD STRATEGY
    def evaluate_theta(self, sol_theta, test_me, test_mom, test_btm, test_return, optimized_nn):
        """
        Evaluate theta optimized return on test sample.

        Parameters
        ----------

        sol_theta: numpy.array
            Best theta founded on optimization.
        test_me: pandas.DataFrame
            Test set of the market capitalization characteristic.
        test_mom: pandas.DataFrame
            Test set of the lagged return characteristic.
        test_btm: pandas.DataFrame
            Test set of the book to market ratio characteristic.
        test_return: pandas.DataFrame
            Test set of the returns.
        optimized_nn: 

        Returns
        -------
        ############### NEED TO ADD TONS OF INFO HERE ###############

        benchmark_test_r: float
            Benchmark return on test set.
        benchmark_test_r_std: float
            Benchmark return standard deviation on test set.
        test_r: float
            Optimized return on test set.
        test_r_std: float
            Optimized return standard deviation on test set.
        """

        LOGGER.info("Evaluating theta on test set.")
        #### TESTING CHARACTERISTICS
        self.training = False
        firm_characteristics_test, r_test, time_test, number_of_stocks = self.create_characteristics(test_me, test_mom, test_btm, test_return)

        #### CREATE BENCHMARK FOR TESTING
        w_benchmark_test = create_w_benchmark(number_of_stocks, time_test)

        ### Create NN Variables
        torch_characteristics_test, torch_r_test, torch_benchmark_test  = convert_to_nn_variables(firm_characteristics_test, r_test, w_benchmark_test)
        torch_r_test = torch_r_test[1:]

        ### Calculate NN test return
        w_test_nn = optimized_nn.weights(torch_characteristics_test[:-1]).squeeze(-1)*1/(number_of_stocks) + torch_benchmark_test[:-1]
        # w_test_nn = (w_test_nn.T/w_test_nn.sum(axis=1)).T

        w_test_nn_constrained = torch.Tensor(constrain_weights(w_test_nn.detach().numpy().T).T)
        w_test_nn_constrained_transaction = w_test_nn_constrained*torch.Tensor(self.test_market_cost[:-1].to_numpy())
        
        self.weights_computed['nn_optimized'].append(w_test_nn.detach().numpy().T)
        self.weights_computed['nn_optimized_constrained'].append(w_test_nn_constrained.detach().numpy().T)

        test_r_nn_sequence = torch.sum((w_test_nn*torch_r_test), dim=1)
        self.test_r_nn = torch.mean(test_r_nn_sequence).detach().numpy()
        self.test_r_nn_std = torch.std(test_r_nn_sequence).detach().numpy()
        test_r_nn_constrained_sequence = torch.sum((w_test_nn_constrained*torch_r_test), dim=1)
        self.test_r_nn_constrained = torch.mean(test_r_nn_constrained_sequence).detach().numpy()
        self.test_r_nn_constrained_std = torch.std(test_r_nn_constrained_sequence).detach().numpy()
        self.test_r_nn_constrained_transaction = torch.mean(torch.sum(w_test_nn_constrained_transaction*torch_r_test, dim=1)).detach().numpy()
        self.test_r_nn_constrained_transaction_std = torch.std(torch.sum(w_test_nn_constrained_transaction*torch_r_test, dim=1)).detach().numpy()

        # leverage_mask = w_test_nn<0
        # leverage = (w_test_nn*leverage_mask).sum(axis=1)
        # min_values, min_idxs = w_test_nn.min(axis=1)
        # max_values, max_idxs = w_test_nn.max(axis=1)

        # Get weight from t and return from t+1
        benchmark_r_test_series=pd.Series(np.sum(w_benchmark_test[:,:-1]*r_test[:,1:], axis=0)).describe()
        benchmark_test_r = benchmark_r_test_series['mean']
        benchmark_test_r_std = benchmark_r_test_series['std']

        ### CREATE TEST WEIGHT AND FIND ITS RETURN
        w_test = np.empty(shape=(number_of_stocks, time_test))
        for i in range(number_of_stocks):
            firm_df = firm_characteristics_test[i].copy()
            firms_coeff = sol_theta.dot(firm_df.T)
            w_test[i] = w_benchmark_test[i] + (1/number_of_stocks)*firms_coeff
        
        # w_test = w_test/w_test.sum(axis=0)
        
        self.weights_computed['optimized'].append(w_test)
        
        w_test_constrained = constrain_weights(w_test.copy())
        self.weights_computed['optimized_constrained'].append(w_test_constrained)

        # Get weight from t and return from t+1
        r_test_sequence = pd.Series(np.sum(w_test[:,:-1]*r_test[:,1:], axis=0))
        r_test_series = r_test_sequence.describe()

        r_test_constrained_sequence = pd.Series(np.sum(w_test_constrained[:,:-1]*r_test[:,1:], axis=0))
        r_test_constrained_series = r_test_constrained_sequence.describe()
        test_r_constrained_transaction_series = pd.Series(np.sum(w_test_constrained[:,:-1]*r_test[:,1:]*self.test_market_cost[:-1].T, axis=0)).describe()

        test_r = r_test_series['mean']
        test_r_std = r_test_series['std']

        test_r_constrained = r_test_constrained_series['mean']
        test_r_constrained_std = r_test_constrained_series['std']

        test_r_constrained_transaction = test_r_constrained_transaction_series['mean']
        test_r_constrained_transaction_std = test_r_constrained_transaction_series['std']

        test_cdi_return = self.cdi_return[-(time_test-1):].reset_index()['Taxa SELIC']
        self.test_cdi_return = test_cdi_return
        test_ibov_return = self.ibov_return[-(time_test-1):].reset_index()['Var%']
        self.test_ibov_return = test_ibov_return

        self.results_comparison["nn_cdi_comparison"] = ((test_r_nn_sequence.detach().numpy() / test_cdi_return)-1)
        self.results_comparison["nn_constraint_cdi_comparison"] = ((test_r_nn_constrained_sequence.detach().numpy() / test_cdi_return)-1)*100
        self.results_comparison["nn_ibov_comparison"] = ((test_r_nn_sequence.detach().numpy() / test_ibov_return)-1)*100
        self.results_comparison["nn_constraint_ibov_comparison"] = ((test_r_nn_constrained_sequence.detach().numpy() - test_ibov_return)-1)*100

        self.results_comparison["opt_cdi_comparison"] = ((r_test_sequence / test_cdi_return)-1)*100
        self.results_comparison["opt_constraint_cdi_comparison"] = ((r_test_constrained_sequence / test_cdi_return)-1)*100
        self.results_comparison["opt_ibov_comparison"] = ((r_test_sequence / test_ibov_return)-1)*100
        self.results_comparison["opt_constraint_ibov_comparison"] = ((r_test_constrained_sequence / test_ibov_return)-1)*100


        test_size = self.results_comparison["nn_cdi_comparison"].shape[0]

        # import pdb; pdb.set_trace()

        df = pd.DataFrame(test_r_nn_constrained_sequence.unsqueeze(0).detach().numpy(), columns=self.timestamp[-test_size:])
        df = df.append(
            pd.DataFrame(
                self.results_comparison["nn_constraint_cdi_comparison"].to_numpy().reshape(1,40),columns =self.timestamp[-test_size:]),
                ignore_index=True 
                )
        df = df.append(
            pd.DataFrame(
                self.results_comparison["nn_constraint_ibov_comparison"].to_numpy().reshape(1,40),columns =self.timestamp[-test_size:]),
                ignore_index=True 
                )
        df['type'] = ["raw_nn_return", "raw_cdi_comparison", "raw_ibov_comparison"]
        df.round(3).set_index('type').to_csv('nn_cdi_ibov_comparison.csv')


        df = pd.DataFrame(r_test_constrained_sequence.to_numpy().reshape(1,40), columns=self.timestamp[-test_size:])
        df = df.append(
            pd.DataFrame(
                self.results_comparison["opt_constraint_cdi_comparison"].to_numpy().reshape(1,40),columns =self.timestamp[-test_size:]),
                ignore_index=True 
                )
        df = df.append(
            pd.DataFrame(
                self.results_comparison["opt_constraint_ibov_comparison"].to_numpy().reshape(1,40),columns =self.timestamp[-test_size:]),
                ignore_index=True 
                )
        df['type'] = ["raw_opt_return", "raw_cdi_comparison", "raw_ibov_comparison"]
        df.round(3).set_index('type').to_csv('opt_cdi_ibov_comparison.csv')

        # import pdb; pdb.set_trace()
    
        # Saving return variables into properties
        self.benchmark_test_r = benchmark_test_r
        self.benchmark_test_r_std  = benchmark_test_r_std

        self.test_r = test_r
        self.test_r_std = test_r_std

        self.test_r_constrained = test_r_constrained
        self.test_r_constrained_std = test_r_constrained_std

        self.test_r_constrained_transaction = test_r_constrained_transaction
        self.test_r_constrained_transaction_std = test_r_constrained_transaction_std

        LOGGER.info("Evalueted theta on test set.")

    
    # TO-DO: Change it to use any number of characteristics
    def create_experiment(self, indexes_list):
        """
        Create experiment using characteristics, list of stocks to look up, 
        indexes of how to split data and the constant of risk aversion.

        Parameters
        ----------

        mcap: pandas.DataFrame
            Market capitalization of the firms
        lreturn: pandas.DataFrame
            Lagged return of the firms.
        book_to_mkt_ratio: pandas.DataFrame
            Book to market ratio of the firms
        monthly_return: pandas.DataFrame
            Monthly return of the firms.
        stocks_names: [str, ]
            List of stock names as strings.
        indexes_list: [( [int, ], [int, ] )]
            List of tuples containing training indexes and testing indexes.
        risk_constant: int
            Risk constant, increase it to become more risk averse.

        Returns
        -------

        benchmark_mean_return: float
            Mean benchmark return for each time frame, in this case for each month.
        mean_obj_r: [float, ]
            List of objective values through each optimization step.
        mean_r: [float, ]
            List of return using optimized weights through each optimization step.
        benchmark_test_r: float
            Benchmark return on test set.
        benchmark_test_r_std: float
            Benchmark return standard deviation on test set.
        test_r: float
            Optimized return on test set.
        test_r_std: float
            Optimized return standard deviation on test set.

        """
        LOGGER.info("Started experiment.")
        np.random.seed(123)
        for train, val, test in indexes_list:
            LOGGER.info("Splitting data into train and test set.")
            theta0 = np.random.rand(1, 3)
            
            train_btm = self.book_to_mkt_ratio.loc[train]
            train_me = self.mcap.loc[train]
            train_mom = self.lreturn.loc[train]
            train_return = self.monthly_return.loc[train]
            self.train_market_cost = self.market_cost.loc[train]

            val_btm = self.book_to_mkt_ratio.loc[val]
            val_me = self.mcap.loc[val]
            val_mom = self.lreturn.loc[val]
            val_return = self.monthly_return.loc[val]
            self.val_market_cost = self.market_cost.loc[val]
            
            test_btm = self.book_to_mkt_ratio.loc[test]
            test_me = self.mcap.loc[test]
            test_mom = self.lreturn.loc[test]
            test_return = self.monthly_return.loc[test]
            self.test_market_cost = self.market_cost.loc[test]

            #### TRAINING CHARACTERISTICS
            self.training = True
            firm_characteristics, r, time, number_of_stocks = self.create_characteristics(train_me, train_mom, train_btm, train_return)
            self.training = False
            firm_characteristics_val, r_val, time_val, number_of_stocks = self.create_characteristics(val_me, val_mom, val_btm, val_return)

            ### Creating weights to a benchmark portifolio using uniform weighted returns
            w_benchmark = create_w_benchmark(number_of_stocks, time)
            w_benchmark_val = create_w_benchmark(number_of_stocks, time_val)

            torch_characteristics, torch_r, torch_benchmark  = convert_to_nn_variables(firm_characteristics, r, w_benchmark)
            torch_characteristics_val, torch_r_val, torch_benchmark_val  = convert_to_nn_variables(
                firm_characteristics_val, r_val, w_benchmark_val)

            ## Using it to be able to set those parameters inside hyperparameter optmization
            self.torch_characteristics = torch_characteristics
            self.torch_r = torch_r
            self.torch_benchmark = torch_benchmark
            self.number_of_stocks = number_of_stocks
            self.torch_characteristics_val = torch_characteristics_val
            self.torch_r_val = torch_r_val 
            self.torch_benchmark_val = torch_benchmark_val

            best_config = self.get_best_hyperparameter_config()

            # best_config = {
            #     'learning_rate':self.learning_rate,
            #     'l2_regularization':self.l2_regularization,
            #     'epochs_size':self.epochs_size,
            #     'adam_betha1':0.9,
            #     'adam_betha2':0.999,
            #     'patience':self.patience,
            # }

            optimized_nn = self.nn_optimizer(
                torch_characteristics, torch_r, torch_benchmark, number_of_stocks, torch_characteristics_val, torch_r_val, torch_benchmark_val, best_config)

            # ### CREATING RETURNS TO COMPARE, OPTIMIZATION STEP
            benchmark_mean_return = np.mean(np.sum(w_benchmark[:,:-1]*r[:,1:], axis=0))
            
            # Creating optimizing solution sol
            self.optimizing_step(firm_characteristics, r, time, number_of_stocks, theta0, w_benchmark, firm_characteristics_val, r_val, time_val, w_benchmark_val)

            sol_theta = self.sol.x
            # sol_theta = ''


            ### Evaluate founded theta on test samples and find its mean return from optimized and benchmark.
            self.evaluate_theta(sol_theta, test_me, test_mom, test_btm, test_return, optimized_nn)
            
            # Optimization step
            self.mean_obj_r_runs.append(self.mean_obj_r)
            self.mean_obj_r_val_runs.append(self.mean_obj_r_val)
            self.mean_r_runs.append(self.mean_r)
            self.mean_constrained_r_runs.append(self.mean_constrained_r)
            self.mean_constrained_transaction_r_runs.append(self.mean_constrained_transaction_r)
            self.benchmark_mean_return_runs.append(benchmark_mean_return)

            # Optimization NN step

            self.nn_loss_runs.append(self.nn_loss)
            self.nn_return_runs.append(self.nn_return)
            self.nn_return_runs_std.append(self.nn_return_std)

            self.nn_val_loss_runs.append(self.nn_val_loss)
            self.nn_val_return_runs.append(self.nn_val_return )
            self.nn_val_return_runs_std.append(self.nn_val_return_std)

            # Test step
            self.benchmark_test_r_runs.append(self.benchmark_test_r)
            self.benchmark_test_r_runs_std.append(self.benchmark_test_r_std)

            self.test_r_runs.append(self.test_r)
            self.test_r_runs_std.append(self.test_r_std)

            self.test_r_constrained_runs.append(self.test_r_constrained) 
            self.test_r_constrained_runs_std.append(self.test_r_constrained_std)

            self.test_r_constrained_transaction_runs.append(self.test_r_constrained_transaction)
            self.test_r_constrained_transaction_runs_std.append(self.test_r_constrained_transaction_std)

            # Test step NN

            self.test_r_nn_runs.append(self.test_r_nn)
            self.test_r_nn_runs_std.append(self.test_r_nn_std)

            self.test_r_nn_constrained_runs.append(self.test_r_nn_constrained)
            self.test_r_nn_constrained_runs_std.append(self.test_r_nn_constrained_std)


            self.test_r_nn_constrained_transaction_runs.append(self.test_r_nn_constrained_transaction)
            self.test_r_nn_constrained_transaction_runs_std.append(self.test_r_nn_constrained_transaction_std)

            LOGGER.info("Finished experiment.")
    

    def plot_final_results(self, experiment_label):
        """
        Plot returns through optimization and in benchmark test.
        """
        # Train results
        mean_obj_r_runs=self.mean_obj_r_runs
        benchmark_mean_return_runs=self.benchmark_mean_return_runs
        mean_r_runs=self.mean_r_runs
        mean_constrained_r_runs=self.mean_constrained_r_runs
        mean_constrained_transaction_r_runs=self.mean_constrained_transaction_r_runs

        # Test results
        benchmark_test_r_mean = np.mean(self.benchmark_test_r_runs, axis=0)
        benchmark_test_r_mean_std= np.mean(self.benchmark_test_r_runs_std, axis=0)
        test_r_mean= np.mean(self.test_r_runs, axis=0)
        test_r_mean_std= np.mean(self.test_r_runs_std, axis=0)
        test_r_constrained_mean = np.mean(self.test_r_constrained_runs, axis=0)
        test_r_constrained_mean_std= np.mean(self.test_r_constrained_runs_std, axis=0)
        test_r_constrained_transaction_mean = np.mean(self.test_r_constrained_transaction_runs, axis=0)
        test_r_constrained_transaction_mean_std = np.mean(self.test_r_constrained_transaction_runs_std, axis=0)
        
        test_r_nn = np.mean(self.test_r_nn_runs, axis=0)
        test_r_nn_constrained = np.mean(self.test_r_nn_constrained_runs, axis=0)
        test_r_nn_constrained_transaction = np.mean(self.test_r_nn_constrained_transaction_runs, axis=0)

        # Sharp ratio
        self.sharp_ratio = np.array(self.test_r_runs)/np.array(self.test_r_runs_std)
        self.sharp_ratio_constrained = np.array(self.test_r_constrained_runs)/np.array(self.test_r_constrained_runs_std)
    
        self.sharp_ratio_nn = np.array(self.test_r_nn_runs)/np.array(self.test_r_nn_runs_std)
        self.sharp_ratio_constrained_nn = np.array(self.test_r_nn_constrained_runs)/np.array(self.test_r_nn_constrained_runs_std)

        sharp_ratio = self.sharp_ratio
        sharp_ratio_constrained = self.sharp_ratio_constrained

        sharp_ratio_nn = self.sharp_ratio_nn
        sharp_ratio_constrained_nn = self.sharp_ratio_constrained_nn

        LOGGER.info("Plotting final results.")
        # Only allows 10 runs, to allow more increase color names.
        colors={
            0:'black',
            1:'blue',
            2:'coral',
            3:'magenta',
            4:'grey',
            5:'violet',
            6:'brown',
            7:'red',
            8:'salmon',
            9:'green'
        }

        ### TRAIN PLOTS ###

        plt.figure(figsize=(12,9))
        plt.title("Mean objective return for each optimization step")
        for run, mean_obj_r in enumerate(mean_obj_r_runs):
            x = range(len(mean_obj_r))
            plt.plot(x, mean_obj_r, label=f'Objective return, Run:{run+1}', c=colors[run])
            plt.plot(x, self.mean_obj_r_val_runs[run], label=f'Objective validation return, Run:{run+1}', c=colors[run+1])
        plt.xlabel('Iteration step')
        plt.ylabel('Objective return')
        plt.legend()
        plt.grid()
        plt.savefig(f'./{experiment_label}_objective_return_over_steps.jpg')
        
        plt.close()

        plt.figure(figsize=(12,9))
        plt.title("Mean return using weight for each optimization step")
        for run, mean_r in enumerate(mean_r_runs):
            x = range(len(mean_r))
            plt.plot(x, mean_r, label=f'Optimized return, Run:{run+1}', c=colors[run])
            plt.plot(x, [benchmark_mean_return_runs[run]]*len(mean_r), label=f'Benchmark return, Run:{run+1}', c=colors[run], linestyle='dashed')
            plt.plot(x, mean_constrained_r_runs[run], label=f'Optimized return with weight constraints, Run:{run+1}', c=colors[run], linestyle='dotted')
            plt.plot(x, mean_constrained_transaction_r_runs[run], label=f'Optimized return with weight constraints and transaction costs, Run:{run+1}', c=colors[run], linestyle='dashdot')
        plt.xlabel('Iteration step')
        plt.ylabel('Mean return')
        plt.legend()
        plt.grid()
        plt.savefig(f'./{experiment_label}_mean_return_over_steps.jpg')
        
        plt.close()
        

        plt.figure(figsize=(12,9))
        plt.title("Train return on NN")
        for run, mean_r in enumerate(self.nn_return_runs):
            x = range(len(mean_r))
            plt.plot(x, mean_r, label=f'Optimized return, Run:{run+1}', c=colors[run])
            plt.plot(x, self.nn_val_return_runs[run], label=f'Optimized validation return, Run:{run+1}', c=colors[run+1])
        plt.ylabel("Mean return")
        plt.xlabel("Epochs")
        plt.legend()
        plt.grid()
        plt.savefig(f'./{experiment_label}_mean_return_over_epochs.jpg')

        plt.close()

        plt.figure(figsize=(12,9))
        plt.title("Loss for each epoch")
        for run, loss in enumerate(self.nn_loss_runs):
            x = range(len(loss))
            plt.plot(x, loss, label=f'Objective loss, Run:{run+1}', c=colors[run])
            plt.plot(x, self.nn_val_loss_runs[run], label=f'Objective validation loss, Run:{run+1}', c=colors[run+1])
        plt.ylabel("Objective loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.grid()
        plt.savefig(f'./{experiment_label}_loss_over_epochs.jpg')

        ################################################
        plt.close()
        ### TEST PLOTS ###
        
        width = 0.1
        br1 = np.arange(1)
        br2 = [x + width for x in br1]
        br3 = [x + width for x in br2]
        br4 = [x + width for x in br3]
        br5 = [x + width for x in br4]
        br6 = [x + width for x in br5]
        br7 = [x + width for x in br6]
        br8 = [x + width for x in br7]
        br9 = [x + width for x in br8]
        plt.figure(figsize=(12,9))
        plt.title("Mean return on test set with standard deviation.")
        plt.ylabel('Mean return')
        plt.bar(br1, test_r_mean, label='Optimized test return', yerr=test_r_mean_std, color='blue', width = 0.09)
        plt.bar(br2, test_r_constrained_mean, label='Optimized test return with weight constraints', yerr=test_r_constrained_mean_std, color='green', width = 0.09)
        plt.bar(br3, test_r_constrained_transaction_mean, label='Optimized test return with weight constraints and transaction costs.', yerr=test_r_constrained_transaction_mean_std, color='black', width = 0.09)
        plt.bar(br4, benchmark_test_r_mean, label='Benchmark test return', yerr=benchmark_test_r_mean_std, color='red', width = 0.09)
        plt.xticks([])
        plt.grid()
        plt.legend()
        plt.savefig(f'./{experiment_label}_mean_return_test_set_with_std.jpg')
        
        plt.close()

        plt.figure(figsize=(12,9))
        plt.title("Mean return on test set")
        plt.ylabel('Mean return')
        plt.bar(br1, test_r_mean, label='Optimized test return',color='blue', width = 0.09, alpha=0.5)
        plt.bar(br2, test_r_constrained_mean, label='Optimized test return with weight constraints', color='green', width = 0.09, alpha=0.5)
        plt.bar(br3, test_r_constrained_transaction_mean, label='Optimized test return with weight constraints and transaction costs.', color='black', width = 0.09, alpha=0.5)
        plt.bar(br4, benchmark_test_r_mean, label='Benchmark test return', color='red', width = 0.09, alpha=0.5)
        plt.bar(br5, test_r_nn, label='Optimized test return using NN', color='brown', width = 0.09, alpha=0.5)
        plt.bar(br6, test_r_nn_constrained, label='Optimized test return using NN with weight constraints', color='coral', width = 0.09, alpha=0.5)
        plt.bar(br7, test_r_nn_constrained_transaction, label='Optimized test return using NN with weight constraints and transaction costs.', color='violet', width = 0.09, alpha=0.5)
        plt.bar(br8, self.test_cdi_return.mean(), label='CDI/Selic mean return', color='cyan', width = 0.09, alpha=0.5)
        plt.bar(br9, self.test_ibov_return.mean(), label='IBOV mean return', color='lime', width = 0.09, alpha=0.5)

        plt.text(-0.04, test_r_mean*1.01, f'Return :{test_r_mean:.3f}%')
        plt.text(width-0.04, test_r_constrained_mean*1.01, f'Return  :{test_r_constrained_mean:.3f}%')
        plt.text(2*width-0.04, test_r_constrained_transaction_mean*1.01, f' Return :{test_r_constrained_transaction_mean:.3f}%')
        plt.text(3*width-0.04 , benchmark_test_r_mean*1.01, f'Return :{benchmark_test_r_mean:.3f}%')
        plt.text(4*width-0.04 , test_r_nn*1.01, f'Return :{test_r_nn:.3f}%')
        plt.text(5*width-0.04 , test_r_nn_constrained*1.01, f'Return :{test_r_nn_constrained:.3f}%')
        plt.text(6*width-0.04 , test_r_nn_constrained_transaction*1.01, f'Return :{test_r_nn_constrained_transaction:.3f}%')
        plt.text(7*width-0.04 , self.test_cdi_return.mean()*1.01, f'Return :{self.test_cdi_return.mean():.3f}%')
        plt.text(8*width-0.04 , self.test_ibov_return.mean()*1.01, f'Return :{self.test_ibov_return.mean():.3f}%')
        plt.xticks([])
        plt.legend(loc="center left")
        plt.savefig(f'./{experiment_label}_mean_return_test_set.jpg')


        if self.plot_weights:
            runs = len(self.weights_computed['optimized'])
            for index in range(runs):
                for weight_type in self.weights_computed:
                    weights = self.weights_computed[weight_type][index]
                    plt.figure(figsize=(12,9))
                    plt.title(f"Stocks {weight_type.replace('_', ' ')} weights over time heatmap")
                    plt.pcolor(weights.T, cmap=cm.seismic)
                    plt.ylabel("Time")
                    plt.xlabel("Stocks")
                    plt.colorbar().set_label("Weights")
                    plt.savefig(f'./{experiment_label}_stock_{weight_type}_weights_run_{index+1}.jpg')
        plt.close()

        plt.figure(figsize=(12,9))
        sharp_ratio = np.mean(sharp_ratio)
        sharp_ratio_constrained = np.mean(sharp_ratio_constrained)
        sharp_ratio_nn=np.mean(sharp_ratio_nn)
        sharp_ratio_constrained_nn=np.mean(sharp_ratio_constrained_nn)
        
        plt.bar(0, sharp_ratio, label='Optimized mean sharp ratio')
        plt.bar(1, sharp_ratio_constrained, label='Optimized constrained mean sharp ratio')
        plt.bar(2, sharp_ratio_nn, label='Optimized mean sharp ratio with NN')
        plt.bar(3, sharp_ratio_constrained_nn, label='Optimized constrained mean sharp ratio with NN')

        plt.text(-0.3, sharp_ratio*1.01, f'Sharp Ratio: {sharp_ratio:.3f}')
        plt.text(1-0.3, sharp_ratio_constrained*1.01, f'Sharp Ratio: {sharp_ratio_constrained:.3f}')
        plt.text(2-0.3, sharp_ratio_nn*1.01, f'Sharp Ratio: {sharp_ratio_nn:.3f}')
        plt.text(3-0.3, sharp_ratio_constrained_nn*1.01, f'Sharp Ratio: {sharp_ratio_constrained_nn:.3f}')
        plt.legend(loc="lower right")
        plt.savefig(f'./{experiment_label}_mean_sharp_ratios.jpg')

        plt.close()

        ##### LEVERAGE PLOTS ##### -> only using first data to be fast
        weight_types = ['optimized', 'nn_optimized']
        stocks_names = np.array(self.stocks_names)
        for weight_type in weight_types:
            time_range = self.weights_computed[weight_type][0].shape[1]
            w = self.weights_computed[weight_type][0]
            plt.figure(figsize=(12,9))
            plt.title(f"Total Leverage percentage for each test time step using {weight_type.replace('_', ' ')} weights")
            plt.ylabel("Percentage")
            plt.xlabel("Test time step")
            plt.bar(range(time_range) ,abs(w*(w<0)).sum(axis=0)*100)
            plt.savefig(f'./{experiment_label}_leverage_{weight_type}.jpg')

            
            min_leverage_values = abs((w*(w<0)).min(axis=0))*100
            plt.figure(figsize=(12,9))
            plt.title(f"Highest leverage percentage for each test time step using {weight_type.replace('_', ' ')} weights")
            plt.ylabel("Percentage")
            plt.xlabel("Test time step")
            plt.bar(range(time_range) , min_leverage_values)
            for i in range(time_range):
                plt.text(i-0.4, min_leverage_values[i]*1.01, stocks_names[i], fontsize=8)
            plt.savefig(f'./{experiment_label}_highest_leverage_{weight_type}.jpg')
        plt.close()
        

        #### COMPARISON WITH IBOV AND SELIC/CDI ######
        comparison_fields = ["cdi", "ibov"]
        calculated_fields = ["nn", "nn_constraint", "opt", "opt_constraint"]

        for comp_field in comparison_fields:
            for cal_field in calculated_fields:
                plt.figure(figsize=(12,9))
                plt.title(f"Comparisson between {cal_field} and {comp_field} results.")
                y = self.results_comparison[f"{cal_field}_{comp_field}_comparison"]
                x = range(len(y))
                plt.ylabel("Difference between return")
                plt.xlabel("Month on test set")
                plt.bar(x, y)
                plt.grid()
                plt.savefig(f'./{experiment_label}_comparison_{cal_field}_{comp_field}.jpg')
        
        LOGGER.info("Saved plots in folder.")



    def _start(self):
        self.load_data()

        self.stocks_names = list(self.monthly_return.columns)
        total_size = self.monthly_return.shape[0]

        indexes_list = []
        for train_percentage, val_percentage in zip(self.train_split, self.val_split):
            idxs_list, = data_split(total_size, train_percentage, val_percentage)
            indexes_list.append(idxs_list)
        
        self.create_experiment(indexes_list)

        experiment_label = "testing_comparison"
        self.plot_final_results(experiment_label)
        LOGGER.info("Done")

def main():
    data_path = "../data/"
    risk_constant = 5
    np.random.seed(123)
    train_split = [0.6]
    val_split = [0.2]
    # train_split = np.random.rand(10)
    single_holdout = ParametricPortifolio(
        data_path=data_path, risk_constant=risk_constant,
        train_split=train_split, val_split=val_split, learning_rate=0.027971, 
        l2_regularization=7.851760e-08, epochs_size=2000, patience=5, plot_weights=True
        )
    single_holdout._start()


if __name__ == "__main__":
    main()