import datetime
import warnings
import os

# After the first iteraction the ParametricPortifolio Object
# gets many parameters and ends up larger than 200MB.
# Ray fails to tune a function that has size larger tahn 95MB
# So to avoid this we set the threshold to 2.5GB since we will
# only use some of their parameters and not all of them.
os.environ['FUNCTION_SIZE_ERROR_THRESHOLD'] = f'{2500*1024*1024}'

import logging
import json
import pathlib
import ray
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import time as tm
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import animation
from matplotlib import cm
from ray import tune
from dateutil.relativedelta import relativedelta
from tqdm.notebook import tqdm
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import jensenshannon
from ray.tune.suggest.bayesopt import BayesOptSearch

torch.random.manual_seed(123)
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'

from utils import (
    data_split, create_w_benchmark, plot_splitted_data,
    utility_function, constrain_weights, compute_transaction_costs,
    ParametricPortifolioNN, loss_fn, convert_to_nn_variables, weight_reset,
    calculate_best_validation_technique
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

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "divide by zero encountered in true_divide")


class ParametricPortifolio():
    """
    Parametric portifolio using firm characteristics 
    to adjust portifolio weights.
    """

    def __init__(self, data_path, risk_constant, train_split, val_split, test_split, benchmark_type, plot_weights, plot_heatmap):
        """
        Initialize object with data path, risk constant, 
        the percentage of train split and validation split, 
        benchmark type and if we plot or not weights and heatmap.

        Parameters
        ----------

        data_path: str
            Path where all data is to be found.
        risk_constant: int
            Risk constant, increase it to become more risk averse.
        train_split: float
            Train percentage to split.
        val_split: float
            Validation percentage to split, start counting after 
            training indexes.
        benchmark_type: str
            Benchmark type use 'value_weighted' to use VW portfolio
            for benchmark or 'equally_weighted' for EQ portfolio.
        plot_weights: bool
            Check if user wants to plot weights from test period.
        plot_heatmap: bool
            Check if user wants to plot an animated weight 
            evolution throughout epochs.

        """
        self.data_path = data_path
        self.risk_constant = risk_constant
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.plot_weights = plot_weights

        self.benchmark_type = benchmark_type

        self.plot_heatmap = plot_heatmap

        # 01/01/1995 - HARDCODED
        base = datetime.datetime.fromtimestamp(788925600)
        # 203 months
        self.timestamp = pd.date_range(base, base + relativedelta(months=+202), freq='MS').strftime("%Y-%b").tolist()

        # Train values
        self.mean_obj_r_runs = []
        self.mean_obj_r_val_runs = []
        self.mean_r_runs = []
        self.mean_constrained_r_runs = []
        self.mean_constrained_transaction_r_runs = []
        self.benchmark_mean_return_runs = []
        self.mean_r_val_runs = []

        # Train values nn
        self.nn_loss_runs = []
        self.nn_return_runs = []
        self.nn_return_runs_std = []
        self.nn_return_constrained_runs = []

        self.nn_val_loss_runs = []
        self.nn_val_return_runs = []
        self.nn_val_return_runs_std = []
        self.nn_val_return_constrained_runs = []

        self.best_config_runs = []

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
        self.status_df_runs = []

        # Benchmark returns

        self.test_cdi_return_runs = []
        self.test_cdi_return_std_runs = []

        self.test_ibov_return_runs = []
        self.test_ibov_return_std_runs = []

        # Thetas

        self.nn_optimized_thetas = []
        self.optimized_thetas = []


    # TO-DO: Change it to load any number of characteristics
    def load_data(self):
        """
        Load returns and firms characteristics from path.

        Parameters
        ----------
        self.data_path: str
            The path where data are.

        Returns
        -------
        self.lreturn: pandas.DataFrame
            Lagged return of the firms.

        self.mcap: pandas.DataFrame
            Market capitalization of the firms

        self.book_to_mkt_ratio: pandas.DataFrame
            Book to market ratio of the firms

        self.monthly_return: pandas.DataFrame
            Monthly return of the firms.

        self.market_cost: pandas.DataFrame
            Market cost using market volume

        self.value_weighted: pandas.DataFrame
            Value weighted benchmark using market cap

        self.cdi_return: pandas.DataFrame
            CDI return from data time range

        self.ibov_return: pandas.DataFrame
            IBOV return from data time range
        """
        # lagged_return
        self.lreturn = pd.read_csv(os.path.join(self.data_path, "monthly_lagged_return.csv"))
        self.mcap = pd.read_csv(os.path.join(self.data_path, "monthly_market_cap.csv"))
        self.book_to_mkt_ratio = pd.read_csv(os.path.join(self.data_path, "monthly_book_to_mkt_ratio.csv"))
        self.monthly_return = pd.read_csv(os.path.join(self.data_path, "monthly_return.csv"))
        # fill nan values with values coming from latter days
        self.lreturn.fillna(method='bfill', inplace=True)
        # fill nan values with values coming from prior days
        self.lreturn.fillna(method='ffill', inplace=True)

        # value weighted benchmark is the weighted by the market cap
        self.value_weighted = (self.mcap.T/self.mcap.sum(axis=1)).T

        # market cap characteristic uses log in order to reduce its magnitude
        self.mcap = np.log(self.mcap)
        # if it has negative infinite, means it was 0 so we change it to 0 again
        self.mcap[self.mcap==-np.inf] = 0

        # Same processing from lagged return
        self.mcap.fillna(method='bfill', inplace=True)
        self.mcap.fillna(method='ffill', inplace=True)

        # Add 1 and log to btm characteristic to be similar to Brandt. et al. article
        self.book_to_mkt_ratio = np.log(1+self.book_to_mkt_ratio)
        self.book_to_mkt_ratio[self.book_to_mkt_ratio==-np.inf] = 0
        self.book_to_mkt_ratio.fillna(method='bfill', inplace=True)
        self.book_to_mkt_ratio.fillna(method='ffill', inplace=True)

        # Same processing from lagged return
        self.monthly_return.fillna(method='bfill', inplace=True)
        self.monthly_return.fillna(method='ffill', inplace=True)

        # Compute market cost based on market volume
        self.market_cost = compute_transaction_costs(self.data_path)
        # Load cdi information from data time range
        self.cdi_return = pd.read_csv(os.path.join(self.data_path, "cdi_return.csv"))
        # Load ibov information from data time range
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
        # Normalize within cross-section of stocks
        for name in characteristics_names:
            #Normalize firm characteristics for all stocks
            mean_charac = firm_characteristics.T.loc[(slice(None), name), :].mean()
            std_charac = firm_characteristics.T.loc[(slice(None), name), :].std()

            firm_characteristics.T.loc[(slice(None), name), :] -= mean_charac
            firm_characteristics.T.loc[(slice(None), name), :] /= std_charac
        
        LOGGER.info("Normalized firm characteristics")
        return firm_characteristics
    

    def optimizing_step(
        self, 
        firm_characteristics, r, time, number_of_stocks, theta0, w_benchmark,
        firm_characteristics_val, r_val, time_val, w_benchmark_val):
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

        w_benchmark: numpy.array
            Benchmark weights, used to create optimized weights.

        transaction_cost: float
            Define transaction cost.

        firm_characteristics_val: pandas.DataFrame
            Normalized firm characteristcs dataframe from validation period.

        r_val: numpy.array
            Return of the stocks through time from validation period.

        time_val: int
            Number of time periods this slice of firm characterists are evaluating on validation period.

        w_benchmark_val: numpy.array
            Benchmark weights, used to create optimized weights using validation period.

        self.risk_constant: int
            Risk constant, increase it to become more risk averse.

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
            Firm characteristics transformed to tensor and having shape
            of (characteristics size, time, number of stocks) instead of
            (number of stocks, time, characteristics size).

        self.torch_r: torch.Tensor
            Return of the stocks through training period. Changed shape
            to be (time, number of stocks) instead of original
            (number of stocks, time).

        self.torch_benchmark: torch.Tensor
            Benchmark weights for training period. Using
            shape (time, number of stocks) instead of original
            (number of stocks, time).

        self.number_of_stocks: Int
            Number of stocks to be used in this experiment

        self.torch_characteristics_val: torch.Tensor
            Firm characteristics from validation period
            transformed to tensor . Same shape change 
            from training period applies.

        self.torch_r_val: torch.Tensor
            Return of the stocks through validation period. 
            Same shape change from training period applies.

        self.torch_benchmark_val: torch.Tensor
            Benchmark weights for validation period. Same
            shape change from training period applies.
            
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
        
        # Last epoch results, retrieving it from object parameters.
        # Function nn optmizer saves those results in object parameters.
        mean_loss = self.nn_val_loss[-1]
        mean_return =  self.nn_val_return[-1]
        mean_std = self.nn_val_return_std[-1]

        # Saving objective function from validation
        # as the *mean_loss*. This is done in order to
        # optimize the hyperparameters based in
        # validation period results.
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
        
        # Dict with range of hyperparemters
        # TO-DO: Change it to use a variable from init
        config = {        
            'learning_rate':tune.uniform(0.01, 0.1),
            'l2_regularization':tune.uniform(1e-7, 1e-2),
            'epochs_size':tune.uniform(500, 3000),
            'adam_betha1':tune.uniform(0.85, 0.95),
            'adam_betha2':tune.uniform(0.99, 0.999),
            'patience':tune.uniform(2, 10),
        }

        # Max 10 iterations displayed each time
        reporter = tune.CLIReporter(max_progress_rows=10)

        # Add mean return and sharpe ratio 
        # as parameters to checked
        reporter.add_metric_column("mean_return")
        reporter.add_metric_column("mean_sharpe_ratio")

        # Using baysean optimization
        bayesopt = BayesOptSearch(verbose=0)

        # Using 25 different samples to find the best 
        # hyperparameter configuration
        analysis = tune.run(
            self.nn_hyperparameter_optimizer, config=config, progress_reporter=reporter, search_alg=bayesopt,
            num_samples=25, metric="mean_loss", mode="min", verbose=0
        )

        LOGGER.info("Finished NN hyperparameter optimization.")
        best_config =  analysis.get_best_config(metric='mean_loss', mode='min')
        LOGGER.info(f"Best NN config: {best_config}")
        return best_config

    def nn_optimizer(
        self,
        torch_characteristics, torch_r, torch_benchmark, number_of_stocks,
        torch_characteristics_val, torch_r_val, torch_benchmark_val, config):
        """
        Optimize and fing theta using a neural network. Theta is the weights of the optimized NN.

        Parameters
        ----------
        torch_characteristics: torch.Tensor
            Firm characteristics transformed to tensor and having shape
            of (characteristics size, time, number of stocks) instead of
            (number of stocks, time, characteristics size).

        torch_r: torch.Tensor 
            Return of the stocks through training period. Changed shape
            to be (time, number of stocks) instead of original
            (number of stocks, time).

        torch_benchmark: torch.Tensor
            Benchmark weights for training period. Using
            shape (time, number of stocks) instead of original
            (number of stocks, time).

        number_of_stocks: int
            Number of stocks to be used in this experiment

        torch_characteristics_val: torch.Tensor
            Firm characteristics from validation period
            transformed to tensor . Same shape change 
            from training period applies.

        torch_r_val: torch.Tensor
            Return of the stocks through validation period. 
            Same shape change from training period applies.

        torch_benchmark_val: torch.Tensor
            Benchmark weights for validation period. Same
            shape change from training period applies.

        config: dict
            Dictionary containing best hyperparameter config such as : 
            learning_rate, l2_regularization, epochs_size, adam_betha1, adam_betha2 and patience.
        
        Returns
        -------

        self.nn_weight_list: [ numpy.Array, ]
            Portfolio weight for each epoch
            using training period.

        self.nn_weight_list_constrained: [ numpy.Array, ]
            Portfolio weight with constraints
            for each epoch using training period.

        self.nn_loss: [ float, ]
            Training objective function
            over epochs in training.
        self.nn_return: [ numpy.Array, ]
            Mean return over epochs in
            training period.
        self.nn_return_std: [ numpy.Array, ]
            Standard deviation from return
            over epochs in training period.
        self.nn_return_constrained: [ numpy.Array, ]
            Mean return over epochs using
            constrained weights in training period.

        self.nn_val_loss: [ float, ]
            Training objective function
            over epochs in validation period.
        self.nn_val_return: [ numpy.Array, ]
            Mean return over epochs in
            validation period.
        self.nn_val_return_std: [ numpy.Array, ]
            Standard deviation from return
            over epochs in validation period.
        self.nn_val_return_constrained: [ numpy.Array, ]
            Mean return over epochs using
            constrained weights in validation period.

        optimized_nn: utils.ParametricPortifolioNN
            Neural network object trained with 
            best found theta ( weights ).

        """
        LOGGER.info("Start neural network optimization.")
        portifolio = ParametricPortifolioNN(torch_benchmark[:-1], torch_r[1:], self.risk_constant, number_of_stocks)
        portifolio_val = ParametricPortifolioNN(torch_benchmark_val[:-1], torch_r_val[1:], self.risk_constant, number_of_stocks)
        portifolio.apply(weight_reset)

        learning_rate = config['learning_rate']
        l2_regularization = config['l2_regularization']
        epochs_size = config['epochs_size']
        # Due to baysean optimization *epochs_size* is a
        # float parameter so we need to convert it to int
        epochs_size = int(epochs_size)
        adam_betha1 = config['adam_betha1']
        adam_betha2 = config['adam_betha2']
        patience = config['patience']
        # Due to baysean optimization *patience* is a
        # float parameter so we need to convert it to int
        patience = int(patience)

        torch_r_val = torch_r_val[1:]
        
        opt = torch.optim.Adam(portifolio.parameters(), betas=(adam_betha1, adam_betha2) , lr=learning_rate, weight_decay=l2_regularization)

        patience_counter = 0
        min_loss_val = 0
        loss_values = []
        return_values_mean = []
        return_values_mean_constrained = []
        return_values_std = []
        return_values_std_constrained = []

        loss_values_val =[]
        return_values_mean_val = []
        return_values_mean_val_constrained = []
        return_values_std_val = []
        return_values_std_val_constrained = []
        self.nn_weight_list = []
        self.nn_weight_list_constrained = []
        for i in range(epochs_size):
            opt.zero_grad()
            # Training period
            value, r_ = portifolio(torch_characteristics[:-1])
            loss = loss_fn(value)
            loss_values.append(loss.item())
            # r_p = torch.sum(r_,-1)

            # Create portfolio weights using theta (or weights) from nn named portifolio
            w_train_nn = portifolio.weights(torch_characteristics[:-1]).squeeze(-1)*1/(number_of_stocks) + torch_benchmark[:-1]
            w_train_nn_constrained = torch.Tensor(constrain_weights(w_train_nn.detach().numpy().T).T)

            # Save training constrained and normal weights
            self.nn_weight_list.append(w_train_nn.detach().numpy())
            self.nn_weight_list_constrained.append(w_train_nn_constrained.detach().numpy())
            
            # Get return from normal and constrained portifolio weight verions 
            r_p = torch.sum((w_train_nn*torch_r[1:]), dim=1)
            r_p_constrained = torch.sum((w_train_nn_constrained*torch_r[1:]), dim=1)

            mean_r_p = torch.mean(r_p).detach().numpy()
            std_r_p = torch.std(r_p).detach().numpy()
            mean_r_p_constrained = torch.mean(r_p_constrained).detach().numpy()
            std_r_p_constrained = torch.std(r_p_constrained).detach().numpy()


            # Validate results on validation period using theta ( or weights ) from training NN
            portifolio_val.weights = portifolio.weights
            value_val, r_val = portifolio_val(torch_characteristics_val[:-1])
            loss_val = loss_fn(value_val)
            # r_p_val = torch.sum(r_val,-1)

            # Same process from training period
            w_val_nn = portifolio_val.weights(torch_characteristics_val[:-1]).squeeze(-1)*1/(number_of_stocks) + torch_benchmark_val[:-1]
            w_val_nn_constrained = torch.Tensor(constrain_weights(w_val_nn.detach().numpy().T).T)
            r_p_val = torch.sum((w_val_nn*torch_r_val), dim=1)
            r_p_val_constrained = torch.sum((w_val_nn_constrained*torch_r_val), dim=1)


            mean_r_p_val = torch.mean(r_p_val).detach().numpy()
            std_r_p_val = torch.std(r_p_val).detach().numpy()
            mean_r_p_val_constrained = torch.mean(r_p_val_constrained).detach().numpy()
            std_r_p_val_constrained = torch.std(r_p_val_constrained).detach().numpy()


            # Save results found for each epoch
            return_values_mean_val.append(mean_r_p_val)
            return_values_std_val.append(std_r_p_val)
            return_values_mean_val_constrained.append(mean_r_p_val_constrained)
            return_values_std_val_constrained.append(std_r_p_val_constrained)
            loss_values_val.append(loss_val.item())
                
            
            if i%100==0:
                theta = portifolio.weights.state_dict()['weight'].detach().numpy()
                LOGGER.debug(f"i:{i}, theta i: {theta}, f(theta):{loss.item()},  Return_mean:{mean_r_p}, Return_std:{std_r_p}")

            return_values_mean.append(mean_r_p)
            return_values_std.append(std_r_p)
            return_values_mean_constrained.append(mean_r_p_constrained)
            return_values_std_constrained.append(std_r_p_constrained)
            loss.backward()
            opt.step()

            
            # Early stopping
            if len(loss_values_val)==1:
                last_loss_val = loss_values_val[-1]
                last_loss_train = loss_values[-1]
            else:
                if last_loss_train > loss_values[-1]:
                    training_decreasing = True
                else:
                    training_decreasing = False
                
                if last_loss_val > loss_values_val[-1]:
                    validation_decreasing = True
                else:
                    validation_decreasing = False

                ## Can change it to happen after first time instead
                ## of count how many times it occured
                if training_decreasing and not validation_decreasing:
                    patience_counter += 1

                last_loss_val = loss_values_val[-1]
                last_loss_train = loss_values[-1]

            if patience_counter == patience:
                break
        
        theta = portifolio.weights.state_dict()['weight'].detach().numpy()

        # Save all results found. Use parameters from this experiment object.
        self.nn_loss = loss_values
        self.nn_return = return_values_mean
        self.nn_return_constrained = return_values_mean_constrained
        self.nn_return_std = return_values_std
    
        self.nn_val_loss = loss_values_val
        self.nn_val_return = return_values_mean_val
        self.nn_val_return_constrained = return_values_mean_val_constrained
        self.nn_val_return_std = return_values_std_val
    

        optimized_nn = portifolio
        return optimized_nn

    # TO-DO: Change it to load any number of characteristics
    # BUY-HOLD STRATEGY
    def evaluate_theta(self, sol_theta, test_me, test_mom, test_btm, test_return, optimized_nn, test):
        """
        Evaluate theta optimized return on test sample.
        Evaluate theta (weights) from NN on test sample.
        Create return comparison with CDI and IBOV.
        Calculate statistics from all returns.

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
        optimized_nn: utils.ParametricPortifolioNN
            Neural network object trained with 
            best found theta ( weights ).
        test: [ int, ]
            List of test indexes.
        self.benchmark_type: str
            Parameter to indicate which benchmark to use
            'value_weighted' for VW portifolio or 
            'equally_weighted' for EQ portifolio.

        Returns
        -------

        self.benchmark_test_r: float
            Benchmark return on test set.
        self.benchmark_test_r_std: float
            Benchmark return standard deviation on test set.
        self.test_r: float
            Optimized return on test set.
        self.test_r_std: float
            Optimized return standard deviation on test set.
        self.test_r_constrained: float
            Mean return at test set using constrained weights
        self.test_r_constrained_std: float.
            Standard deviation from return at test 
            set using constrained weights.
        self.test_r_constrained_transaction: float
            Mean return at test set using constrained weights
            and transaction costs.
        self.test_r_constrained_transaction_std: float
            Standard deviation from return at test 
            set using constrained weights and transaction costs.

        self.weights_computed: dict( list(numpy.Array, ) )
            Weights computed from NN and OPT best thetas at
            test period. Also save constrained weight versions.

        self.test_r_nn:
            Mean return on test set using neural networks.
        self.test_r_nn_std:
            Standard deviation from return on test set 
            using neural networks.
        self.test_r_nn_constrained:
            Mean return on test set using neural networks
            and weights constraints.
        self.test_r_nn_constrained_std:
            Standard deviation return on test set using neural 
            networks and weights constraints.
        self.test_r_nn_constrained_transaction:
            Mean return on test set using neural networks, weights 
            constraints and transaction costs.
        self.test_r_nn_constrained_transaction_std:
            Standard deviation return on test set using neural networks, 
            weights constraints and transaction costs.

        self.test_cdi_return:
            CDI return sequence at test set.
        self.test_ibov_return:
            IBOV return sequence at test set.
        """

        LOGGER.info("Evaluating theta on test set.")
        #### TESTING CHARACTERISTICS
        firm_characteristics_test, r_test, time_test, number_of_stocks = self.create_characteristics(test_me, test_mom, test_btm, test_return)

        #### CREATE BENCHMARK FOR TESTING
        if self.benchmark_type == 'value_weighted':
            w_benchmark_test = self.value_weighted.iloc[test].T.to_numpy()
        else:
            w_benchmark_test = create_w_benchmark(number_of_stocks, time_test)

        ### Create NN Variables
        torch_characteristics_test, torch_r_test, torch_benchmark_test  = convert_to_nn_variables(firm_characteristics_test, r_test, w_benchmark_test)
        old_torch_r_test = torch_r_test.clone()
        torch_r_test = torch_r_test[1:]

        ### Calculate NN test return
        w_test_nn = optimized_nn.weights(torch_characteristics_test[:-1]).squeeze(-1)*1/(number_of_stocks) + torch_benchmark_test[:-1]
        # w_test_nn = (w_test_nn.T/w_test_nn.sum(axis=1)).T

        w_test_nn_constrained = torch.Tensor(constrain_weights(w_test_nn.detach().numpy().T).T)
        w_test_nn_constrained_transaction = w_test_nn_constrained*torch.Tensor(self.test_market_cost[:-1].to_numpy())
        
        # Save computed weights on test set from neural network
        self.weights_computed['nn_optimized'].append(w_test_nn.detach().numpy().T)
        self.weights_computed['nn_optimized_constrained'].append(w_test_nn_constrained.detach().numpy().T)

        ## Create nn return sequence to be used after
        test_r_nn_sequence = torch.sum((w_test_nn*torch_r_test), dim=1)
        test_r_nn_constrained_sequence = torch.sum((w_test_nn_constrained*torch_r_test), dim=1)

        # Save all mean and std from returns with constrained and unconstrained weights
        self.test_r_nn = torch.mean(test_r_nn_sequence).detach().numpy()
        self.test_r_nn_std = torch.std(test_r_nn_sequence).detach().numpy()
        self.test_r_nn_constrained = torch.mean(test_r_nn_constrained_sequence).detach().numpy()
        self.test_r_nn_constrained_std = torch.std(test_r_nn_constrained_sequence).detach().numpy()
        self.test_r_nn_constrained_transaction = torch.mean(torch.sum(w_test_nn_constrained_transaction*torch_r_test, dim=1)).detach().numpy()
        self.test_r_nn_constrained_transaction_std = torch.std(torch.sum(w_test_nn_constrained_transaction*torch_r_test, dim=1)).detach().numpy()


        # Get weight from t and return from t+1
        benchmark_r_test_series=pd.Series(np.sum(w_benchmark_test[:,:-1]*r_test[:,1:], axis=0)).describe()
        benchmark_test_r = benchmark_r_test_series['mean']
        benchmark_test_r_std = benchmark_r_test_series['std']

        ### Create test weight from Optimized theta and find its return
        w_test = np.empty(shape=(number_of_stocks, time_test))
        for i in range(number_of_stocks):
            firm_df = firm_characteristics_test[i].copy()
            firms_coeff = sol_theta.dot(firm_df.T)
            w_test[i] = w_benchmark_test[i] + (1/number_of_stocks)*firms_coeff
        
        # w_test = w_test/w_test.sum(axis=0)
        
        # Save computed weights on test set from optimize algorithm
        self.weights_computed['optimized'].append(w_test)
        w_test_constrained = constrain_weights(w_test.copy())
        self.weights_computed['optimized_constrained'].append(w_test_constrained)

        ## Create nn optimized sequence to be used after

        # Get weight from t and return from t+1
        r_test_sequence = pd.Series(np.sum(w_test[:,:-1]*r_test[:,1:], axis=0))
        r_test_series = r_test_sequence.describe()
        
        r_test_constrained_sequence = pd.Series(np.sum(w_test_constrained[:,:-1]*r_test[:,1:], axis=0))
        r_test_constrained_series = r_test_constrained_sequence.describe()
        test_r_constrained_transaction_series = pd.Series(np.sum(w_test_constrained[:,:-1]*r_test[:,1:]*self.test_market_cost[:-1].T, axis=0)).describe()

        test_r = r_test_series['mean']
        test_r_std = r_test_series['std']

        # Create comparison excel to compare IBOV, CDI, OPT and NN returns
        self.create_comparison_excel(
            optimized_nn, torch_characteristics_test, number_of_stocks,
            torch_benchmark_test, old_torch_r_test, sol_theta, firm_characteristics_test, 
            r_test, w_benchmark_test, 'test', test)

        test_r_constrained = r_test_constrained_series['mean']
        test_r_constrained_std = r_test_constrained_series['std']

        test_r_constrained_transaction = test_r_constrained_transaction_series['mean']
        test_r_constrained_transaction_std = test_r_constrained_transaction_series['std']

        test_cdi_return = self.cdi_return[-(time_test):-1].reset_index()['Taxa SELIC']
        test_ibov_return = self.ibov_return[-(time_test):-1].reset_index()['Var%']

        # Save cdi and ibov returns from test period
        self.test_cdi_return = test_cdi_return
        self.test_ibov_return = test_ibov_return

        
        ## Calculate statistics

        ## NN 
        ### Normal
        test_r_nn_sequence = test_r_nn_sequence.detach().numpy()
        w_test_nn = w_test_nn.detach().numpy()
        torch_r_test = torch_r_test.detach().numpy()
        nn_stats_info = self.calculate_statistics(
            w_test_nn, test_r_nn_sequence, 'nn', test_cdi_return, torch_r_test
            )

        nn_df = pd.DataFrame(list(nn_stats_info.values()), columns=['Neural Network'])
        nn_df['Statistic Name'] = list(nn_stats_info.keys())

        ### Constrained
        test_r_nn_constrained_sequence = test_r_nn_constrained_sequence.detach().numpy()
        w_test_nn_constrained = w_test_nn_constrained.detach().numpy()
        nn_constrained_stats_info = self.calculate_statistics(
            w_test_nn_constrained, test_r_nn_constrained_sequence, 
            'nn', test_cdi_return, torch_r_test
            )

        nn_constrained_df = pd.DataFrame(
            list(nn_constrained_stats_info.values()), 
            columns=['Neural Network Constrained']
            )

        ## OPT
        ### Normal
        test_r_opt_sequence = r_test_sequence.to_numpy()
        w_test_opt = w_test[:,:-1]
        r_test_opt = r_test[:,1:]

        opt_stats_info = self.calculate_statistics(
            w_test_opt, test_r_opt_sequence, 'opt', test_cdi_return, r_test_opt
            )
        
        opt_df = pd.DataFrame(list(opt_stats_info.values()), columns=['Optimized'])

        ### Constrained
        test_r_opt_sequence_constrained = r_test_constrained_sequence.to_numpy()
        w_test_opt_constrained = w_test_constrained[:,:-1]
        r_test_opt_constrained = r_test[:,1:]

        opt_constrained_stats_info = self.calculate_statistics(
            w_test_opt_constrained, test_r_opt_sequence_constrained, 
            'opt', test_cdi_return, r_test_opt_constrained
            )

        
        opt_constrained_df = pd.DataFrame(
            list(opt_constrained_stats_info.values()), 
            columns=['Optimized Constrained']
            )

        stats_df = pd.concat([nn_df, nn_constrained_df, opt_df, opt_constrained_df], axis=1).set_index('Statistic Name')

        ## ADD TO DOCUMENTATIO
        self.status_df_runs.append(stats_df)

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

    def calculate_statistics(self, w_test, test_r_sequence, matrix_type, test_cdi_return, r_test):
        """
        Calculate all statistics for each model chosen

        Parameters
        ----------
        w_test: numpy.Array
            Weights from chosen model

        test_r_sequence: numpy.Array
            Return from chosen model

        matrix_type: str
            Set this as 'opt' if using 
            optimizer matrix

        test_cdi_return: numpy.Array 
            Risk free return

        r_test: numpy.Array
            Raw return in specific format

        Returns
        -------
        stats_info: dict
            Dictionary containing multiple 
            statistical information.

        """

        stats_info = {}
        shape_time = 0
        shape_stocks = 1


        if matrix_type=='opt':
            r_test = r_test.T
            w_test = w_test.T
        
        # Calculate STD from returns
        test_std = np.std(test_r_sequence)
        # Calculate new sequence removing its mean
        test_r_sequence_star = test_r_sequence - test_r_sequence.mean()
        # Calculate partial lower STD ( std from negative numbers )
        partial_std = test_r_sequence_star[test_r_sequence_star<0].std()
        # Calculate kurtosis from returns
        test_kurt = kurtosis(test_r_sequence, fisher=True)
        # Calculate skewness from returns
        test_skew = skew(test_r_sequence)
        # Calculate excess return using CDI as risk free return
        excess_return = np.mean(test_r_sequence-test_cdi_return)
        
        # Try to find if there is a time in return sequence
        # that your fund is broken. E.G. there is more than
        # 100% negative return. Skip that time and calculate
        # cumulative return starting from next time.
        return_sequence = ((test_r_sequence/100)+1)
        if (return_sequence<0).any():
            positions = []
            for idx, value in enumerate(return_sequence):
                if value < 0:
                    positions.append(idx)
            start_idx = positions[-1]
        else:
            start_idx = -1
        
        # Calculate cumulative return from return sequence
        cumulative_return = np.cumprod(return_sequence[start_idx+1:])[-1]*100
        # Calculate sharpe ratio from return using excess return
        sharpe_ratio = excess_return/test_std
        
        # Mean of Max weight found at each time
        mean_max = np.mean(np.max(w_test, axis=shape_stocks))
        # Mean of min weight found at each time
        mean_min = np.mean(np.min(w_test, axis=shape_stocks))
        # How much leverage there is at each time
        mean_gross_leverage = np.mean(np.sum(np.abs(w_test), axis=shape_stocks))
        # Mean proportion of negative weights at each time
        proportion_leverage = np.mean(w_test<0)*100


        return_test_hold = np.empty_like(w_test)

        w_r_test = w_test*r_test

        for idx_h_r in range(r_test.shape[shape_time]):
            return_test_hold[idx_h_r] = ((w_r_test[idx_h_r]+100)/100) / ((test_r_sequence[idx_h_r]+100)/100)

        w_test_hold = np.empty_like(w_test)

        for idx_h_w in range(1, r_test.shape[shape_time]):
            w_test_hold[idx_h_w] = return_test_hold[idx_h_w-1] * w_test[idx_h_w-1]

        # Calculate average turnover ( how better would be to hold instead of changing weights )
        avg_turnover= np.abs(w_test[1:]-w_test_hold[1:]).mean(axis=shape_stocks).mean()*100

        total_diversification = []
        
        for idx_div in range(2, w_test.shape[shape_time]+1):
            diversitfication = w_test[idx_div-1].dot(
                w_r_test[:idx_div].std(axis=shape_time)) / np.sqrt(
                    w_test[idx_div-1].dot(
                        np.cov(
                            r_test[:idx_div].T)).dot(
                                w_test[idx_div-1].T
                                )
                            )
            total_diversification.append(diversitfication)
        
        # Calculate total diversitication ( how distributed are your return distribution )
        average_diverstification = np.mean(total_diversification)

        stats_info['Standard Deviation (%)']  = np.round(test_std, 4)
        stats_info['Lower Partial Standard Deviation (%)']  = np.round(partial_std, 4)
        stats_info['Kurtosis']  = np.round(test_kurt, 4)
        stats_info['Skewness']  = np.round(test_skew, 4)
        stats_info['Average Diversification Ratio']  = np.round(average_diverstification, 4)
        stats_info['Average Max. Weight']  = np.round(mean_max, 4)
        stats_info['Average Min. Weight']  = np.round(mean_min, 4)
        stats_info['Average Gross Leverage']  = np.round(mean_gross_leverage, 4)
        stats_info['Proportion of Leverage (%)']  = np.round(proportion_leverage, 4)
        stats_info['Average Turnover (%)']  = np.round(avg_turnover, 4)
        stats_info['Average Excess Return (%)']  = np.round(excess_return, 4)
        stats_info['Cumulative Return (%)']  = np.round(cumulative_return, 4)
        stats_info['Sharpe Ratio']  = np.round(sharpe_ratio, 4)
        return stats_info

    def create_comparison_excel(
        self, optimized_nn, torch_characteristics, number_of_stocks,
        torch_benchmark, torch_r, sol_theta, firm_characteristics, 
        r, w_benchmark, data_type, data_size):
        """
        Create comparison excel using all returns
        found from NN and OPT model with CDI and IBOV benchmarks.

        Parameters
        ----------
        optimized_nn: utils.ParametricPortifolioNN
            Neural network object trained with 
            best found theta ( weights ).
        torch_characteristics: torch.Tensor
            Normalized firm characteristcs dataframe
            reshaped to torch.Tensor with shape
            (characteristics size, time, number of stocks).
        number_of_stocks: int
            Number of stocks to be used in this experiment
        torch_benchmark: torch.Tensor
            Benchmark weights. Using
            shape (time, number of stocks) instead of original
            (number of stocks, time).
        torch_r: torch.Tensor
            Return of the stocks. Changed shape
            to be (time, number of stocks) instead of original
            (number of stocks, time).
        sol_theta: numpy.Array
            Best theta founded on optimization.
        firm_characteristics: pandas.DataFrame
            Normalized firm characteristcs dataframe.
        r: numpy.Array
            Array of returns
        w_benchmark: numpy.Array
            Array of benchmark weights
        data_type: str
            Data period to be compared.
            Use test, train or val.
        data_size: [int, ]
            List of data index to be used

        """
        # Get NN theta (weights) and create portfolio weights
        w_nn = optimized_nn.weights(torch_characteristics[:-1]).squeeze(-1)*1/(number_of_stocks) + torch_benchmark[:-1]
        w_nn_constrained = torch.Tensor(constrain_weights(w_nn.detach().numpy().T).T)
        r_nn_constrained_sequence = torch.sum((w_nn_constrained*torch_r[1:]), dim=1)

        # Get OPT theta and create portfolio weights
        w = np.empty(shape=(number_of_stocks, len(data_size)))
        for i in range(number_of_stocks):
            firm_df = firm_characteristics[i].copy()
            firms_coeff = sol_theta.dot(firm_df.T)
            w[i] = w_benchmark[i] + (1/number_of_stocks)*firms_coeff

        w_constrained = constrain_weights(w.copy())
        r_constrained_sequence = pd.Series(np.sum(w_constrained[:,:-1]*r[:,1:], axis=0))

        data_dimension = len(data_size[1:])

        ## Compare returns with CDI and IBOV for same given time.
        cdi_return = self.cdi_return.loc[data_size[1:]].reset_index()['Taxa SELIC']
        ibov_return = self.ibov_return.loc[data_size[1:]].reset_index()['Var%']
        
        self.results_comparison[f"{data_type}_nn_constraint_cdi_comparison"] = ((r_nn_constrained_sequence.detach().numpy() / cdi_return)-1)*100
        self.results_comparison[f"{data_type}_nn_constraint_ibov_comparison"] = ((r_nn_constrained_sequence.detach().numpy() / ibov_return)-1)*100


        self.results_comparison[f"{data_type}_opt_constraint_cdi_comparison"] = ((r_constrained_sequence / cdi_return)-1)*100
        self.results_comparison[f"{data_type}_opt_constraint_ibov_comparison"] = ((r_constrained_sequence / ibov_return)-1)*100

        lower_idx = data_size[1]
        upper_idx = data_size[-1]+1
        
        df = pd.DataFrame(r_nn_constrained_sequence.unsqueeze(0).detach().numpy(), columns=self.timestamp[lower_idx:upper_idx] )
        df = df.append(
            pd.DataFrame(
                self.results_comparison[f"{data_type}_nn_constraint_cdi_comparison"].to_numpy().reshape(1,data_dimension),
                columns =self.timestamp[lower_idx:upper_idx]),
                ignore_index=True 
                )
        df = df.append(
            pd.DataFrame(
                self.results_comparison[f"{data_type}_nn_constraint_ibov_comparison"].to_numpy().reshape(1,data_dimension),
                columns =self.timestamp[lower_idx:upper_idx]),
                ignore_index=True 
                )
        df['Return type'] = ["Neural Network return (%)", "NN return compared to CDI/Selic (%)", "NN return compared to IBOV(%)"]
        df.round(3).set_index('Return type').to_csv(f'./sheets/{data_type}_nn_cdi_ibov_comparison.csv')


        df = pd.DataFrame(r_constrained_sequence.to_numpy().reshape(1,data_dimension), columns=self.timestamp[lower_idx:upper_idx])
        df = df.append(
            pd.DataFrame(
                self.results_comparison[f"{data_type}_opt_constraint_cdi_comparison"].to_numpy().reshape(1,data_dimension),
                columns=self.timestamp[lower_idx:upper_idx]),
                ignore_index=True 
                )
        df = df.append(
            pd.DataFrame(
                self.results_comparison[f"{data_type}_opt_constraint_ibov_comparison"].to_numpy().reshape(1,data_dimension),
                columns=self.timestamp[lower_idx:upper_idx]),
                ignore_index=True 
                )
        df['Return type'] = ["Optimization return (%)", "Optimization return compared to CDI/Selic (%)", "Optimization return compared to IBOV(%)"]
        df.round(3).set_index('Return type').to_csv(f'./sheets/{data_type}_opt_cdi_ibov_comparison.csv')

    # TO-DO: Change it to use any number of characteristics
    def create_experiment(self, indexes_list):
        """
        Create experiment using characteristics, list of stocks to look up, 
        indexes of how to split data and the constant of risk aversion.

        Parameters
        ----------

        self.mcap: pandas.DataFrame
            Market capitalization of the firms
        self.lreturn: pandas.DataFrame
            Lagged return of the firms.
        self.book_to_mkt_ratio: pandas.DataFrame
            Book to market ratio of the firms
        self.monthly_return: pandas.DataFrame
            Monthly return of the firms.
        self.stocks_names: [str, ]
            List of stock names as strings.
        self.risk_constant: int
            Risk constant, increase it to become more risk averse.
        self.market_cost: numpy.Array
            Array with market cost using stock volume as base.
        self.value_weighted: numpy.Array
            Value weighted benchamrk using market cap.
        indexes_list: [( [int, ], [int, ] )]
            List of tuples containing training indexes and testing indexes.

        Returns
        -------        
        self.train_market_cost: numpy.Array
            Spllited market cost into train period
        self.val_market_cost: numpy.Array
            Spllited market cost into validation period
        self.test_market_cost: numpy.Array
            Spllited market cost into test period

        self.torch_characteristics: torch.Tensor
            Firm characteristics transformed to tensor and having shape
            of (characteristics size, time, number of stocks) instead of
            (number of stocks, time, characteristics size).
        self.torch_r: torch.Tensor
            Return of the stocks through training period. Changed shape
            to be (time, number of stocks) instead of original
            (number of stocks, time).
        self.torch_benchmark: torch.Tensor
            Benchmark weights for training period. Using
            shape (time, number of stocks) instead of original
            (number of stocks, time).

        self.number_of_stocks: int
            Number of stocks to be used in this experiment

        self.torch_characteristics_val: torch.Tensor
            Firm characteristics from validation period
            transformed to tensor . Same shape change 
            from training period applies.

        self.torch_r_val: torch.Tensor
            Return of the stocks through validation period. 
            Same shape change from training period applies.
        self.torch_benchmark_val: torch.Tensor
            Benchmark weights for validation period. Same
            shape change from training period applies.

        # taining parameters
        self.mean_obj_r_runs: [[float, ], ]
            List of a list of objective values from 
            optimized model at training period. Each 
            run produces a list of objective values 
            for each optimization step.

        self.mean_obj_r_val_runs: [[float, ], ]
            Same as mean_obj_r_runs only change 
            time period to validation period.

        self.mean_r_runs: [[float, ], ]
            List of a list of mean return from 
            optimized model at training period. Each 
            run produces a list of mean return 
            for each optimization step.

        self.mean_constrained_r_runs: [[float, ], ]
            List of a list of mean return from 
            optimized model at training period using
            constrained weights. Each run produces a 
            list of mean return for each optimization 
            step.

        self.mean_constrained_transaction_r_runs: [[float, ], ]
            List of a list of mean return from 
            optimized model at training period using
            constrained weights and transaction costs. 
            Each run produces a list of mean return for 
            each optimization step.

        self.benchmark_mean_return_runs: [float, ]
            List of benchmark mean return for each run.
            It is constant throughout all training process.

        self.nn_loss_runs: [[float, ], ]
            List of a list of loss from
            neural network model at traning period.
            Each run produces a list of loss
            for each epoch.
        self.nn_return_runs: [[float, ], ]
            List of a list of mean return from
            neural network model at traning period.
            Each run produces a list of mean return
            for each epoch.
        self.nn_return_runs_std: [[float, ], ]
            List of a list of std from return from
            neural network model at traning period.
            Each run produces a list of std from 
            return for each epoch.

        self.nn_val_loss_runs: [[float, ], ]
            Same as nn_loss_runs, only change
            the time period to validation.
        self.nn_val_return_runs: [[float, ], ]
            Same as nn_return_runs, only change
            the time period to validation.
        self.nn_val_return_runs_std: [[float, ], ]
            Same as nn_return_runs_std, only change
            the time period to validation.

        # test parameters
        self.benchmark_test_r_runs: [float, ]
            Benchmark mean return for test period
            at each run.
        self.benchmark_test_r_runs_std: [float, ]
            Benchmark std from return for test period
            at each run.

        self.test_r_runs: [float, ]
            List of mean return on test set for
            each run with optmization model.
        self.test_r_runs_std: [float, ]
            List of std from return on test set for
            each run with optmization model.
        self.test_r_constrained_runs: [float, ]
            List of mean return on test set using
            constrained weights for each run 
            with optmization model.
        self.test_r_constrained_runs_std: [float, ]
            List of std from return on test set using
            constrained weights for each run and
            with optmization model.
        self.test_r_constrained_transaction_runs: [float, ]
            List of mean return on test set using
            constrained weights and transaction costs 
            for each run with optmization model.
        self.test_r_constrained_transaction_runs_std: [float, ]
            List of std from return on test set using
            constrained weights and transaction costs 
            for each run with optmization model.

        self.test_r_nn_runs: [float, ]
            List of mean return on test set for
            each run with neural network model.            
        self.test_r_nn_runs_std: [float, ]
            List of std from return on test set for
            each run with neural network model.   
        self.test_r_nn_constrained_runs: [float, ]
            List of mean return on test set using
            constrained weights for each run 
            with neural network model.
        self.test_r_nn_constrained_runs_std: [float, ]
            List of std from return on test set using
            constrained weights for each run 
            with neural network model.
        self.test_r_nn_constrained_transaction_runs: [float, ]
            List of mean return on test set using
            constrained weights and transaction costs 
            for each run with neural network model.
        self.test_r_nn_constrained_transaction_runs_std: [float, ]
            List of std from return on test set using
            constrained weights and transaction costs 
            for each run with neural network model.

        """
        LOGGER.info("Started experiment.")
        np.random.seed(123)
        for train, val, test in indexes_list:
            # Getting global values 
            btm = self.book_to_mkt_ratio
            me = self.mcap
            mom = self.lreturn
            return_ = self.monthly_return

            firm_characteristics_global, r_global, time_global, number_of_stocks = self.create_characteristics(
            me, mom, btm, return_
            )

            if self.benchmark_type == 'value_weighted':
                w_benchmark_global = self.value_weighted.T.to_numpy()
            else:
                w_benchmark_global = create_w_benchmark(number_of_stocks, time_global)
                
            torch_characteristics_global, torch_r_global, torch_benchmark_global  = convert_to_nn_variables(
                firm_characteristics_global, r_global, w_benchmark_global
                )
            
            calculate_best_validation_technique(firm_characteristics_global)

            LOGGER.info("Splitting data into train, validation and test set.")
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
            firm_characteristics, r, time, number_of_stocks = self.create_characteristics(train_me, train_mom, train_btm, train_return)
            firm_characteristics_val, r_val, time_val, number_of_stocks = self.create_characteristics(val_me, val_mom, val_btm, val_return)

            ### Creating weights to a benchmark portifolio
            if self.benchmark_type == 'value_weighted':
                w_benchmark = self.value_weighted.iloc[train].T.to_numpy()
                w_benchmark_val = self.value_weighted.iloc[val].T.to_numpy()
            else:
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
            
            ## TO-DO: put inside documentation
            self.best_config_runs.append(best_config)

            optimized_nn = self.nn_optimizer(
                torch_characteristics, torch_r, torch_benchmark, number_of_stocks, torch_characteristics_val, torch_r_val, torch_benchmark_val, best_config)


            # ### CREATING RETURNS TO COMPARE, OPTIMIZATION STEP
            benchmark_mean_return = np.mean(np.sum(w_benchmark[:,:-1]*r[:,1:], axis=0))
            
            # Creating optimizing solution sol
            self.optimizing_step(firm_characteristics, r, time, number_of_stocks, theta0, w_benchmark, firm_characteristics_val, r_val, time_val, w_benchmark_val)

            sol_theta = self.sol.x

            # Compare return from each time to same time in IBOV and CDI
            global_ = np.arange(time_global)
            self.create_comparison_excel(
            optimized_nn, torch_characteristics_global, number_of_stocks,
            torch_benchmark_global, torch_r_global, sol_theta, firm_characteristics_global, 
            r_global, w_benchmark_global, 'global', global_)


            ### Evaluate founded theta on test samples and find its mean return from optimized and nn models.
            self.evaluate_theta(sol_theta, test_me, test_mom, test_btm, test_return, optimized_nn, test)
            
            # Optimization step training values
            self.mean_obj_r_runs.append(self.mean_obj_r)
            self.mean_obj_r_val_runs.append(self.mean_obj_r_val)
            self.mean_r_runs.append(self.mean_r)
            self.mean_constrained_r_runs.append(self.mean_constrained_r)
            self.mean_constrained_transaction_r_runs.append(self.mean_constrained_transaction_r)
            self.mean_r_val_runs.append(self.mean_r_val)
            self.benchmark_mean_return_runs.append(benchmark_mean_return)

            # Optimization NN step training values
            self.nn_loss_runs.append(self.nn_loss)
            self.nn_return_runs.append(self.nn_return)
            self.nn_return_runs_std.append(self.nn_return_std)
            self.nn_return_constrained_runs.append(self.nn_return_constrained)

            self.nn_val_loss_runs.append(self.nn_val_loss)
            self.nn_val_return_runs.append(self.nn_val_return )
            self.nn_val_return_runs_std.append(self.nn_val_return_std)
            self.nn_val_return_constrained_runs.append(self.nn_val_return_constrained)

            # Optimization step test values
            self.benchmark_test_r_runs.append(self.benchmark_test_r)
            self.benchmark_test_r_runs_std.append(self.benchmark_test_r_std)

            self.test_r_runs.append(self.test_r)
            self.test_r_runs_std.append(self.test_r_std)

            self.test_r_constrained_runs.append(self.test_r_constrained) 
            self.test_r_constrained_runs_std.append(self.test_r_constrained_std)

            self.test_r_constrained_transaction_runs.append(self.test_r_constrained_transaction)
            self.test_r_constrained_transaction_runs_std.append(self.test_r_constrained_transaction_std)

            # Optimization NN step test values
            self.test_r_nn_runs.append(self.test_r_nn)
            self.test_r_nn_runs_std.append(self.test_r_nn_std)

            self.test_r_nn_constrained_runs.append(self.test_r_nn_constrained)
            self.test_r_nn_constrained_runs_std.append(self.test_r_nn_constrained_std)

            self.test_r_nn_constrained_transaction_runs.append(self.test_r_nn_constrained_transaction)
            self.test_r_nn_constrained_transaction_runs_std.append(self.test_r_nn_constrained_transaction_std)


            ## Benchmark returns on test values -- ADD TO DOCUMENTATION

            self.test_cdi_return_runs.append(self.test_cdi_return.mean())
            self.test_cdi_return_std_runs.append(self.test_cdi_return.std())

            self.test_ibov_return_runs.append(self.test_ibov_return.mean())
            self.test_ibov_return_std_runs.append(self.test_ibov_return.std())

            # Thetas -- ADD TO DOCUMENTATION

            self.optimized_thetas.append(sol_theta)
            nn_theta = optimized_nn.weights.state_dict()['weight'].detach().numpy()
            self.nn_optimized_thetas.append(nn_theta)


            LOGGER.info("Finished experiment.")
    
    def plot_animated_heatmap(self, weight_list, weight_type, experiment_label):
        """
        Plot animated heatmap from weights over epochs.

        Parameters
        ----------

        weight_list: [numpy.Array, ]
            List of generated weights from each epoch.
        weight_type: str
            String to tell if weights are from nn or nn_constrained.
        experiment_label: str
            Experiment label to be used on saved gif.

        """
        plt.rcParams["figure.figsize"] = [12, 9]
        plt.rcParams["figure.autolayout"] = True

        fig = plt.figure()
        first_weight = weight_list[0]
        dimension = first_weight.shape
        epochs = len(weight_list)//20
        min_w = min([w.min() for w in weight_list])
        max_w = max([w.max() for w in weight_list])
        ax = plt.axes()
        sns.heatmap(first_weight, vmax=max_w, vmin=min_w, ax=ax)
        LOGGER.info(f"Generating animated heatmap from {weight_type} with {epochs} frames")

        def init():
            sns.heatmap(np.zeros(dimension), vmax=max_w, vmin=min_w, cbar=False, ax=ax)

        def animate(i):
            data = weight_list[i*10]
            sns.heatmap(data, vmax=max_w, vmin=min_w, cbar=False, ax=ax)
            ax.set_title(f"Epoch: {i*10}")
        
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=epochs, repeat=False)
        writer = animation.PillowWriter(fps=30)
        anim.save(f'./images/{weight_type}_heatmap_weights_{experiment_label}.gif', writer=writer)

    def plot_final_results(self, experiment_label):
        """
        Plot returns through optimization and in benchmark test.

        Parameters
        ----------

        experiment_label: str

        self.mean_obj_r_runs: [[float, ], ]
            List of a list of objective values from 
            optimized model at training period. Each 
            run produces a list of objective values 
            for each optimization step.
        
        self.mean_obj_r_val_runs: [[float, ], ]
            Same as mean_obj_r_runs only change 
            time period to validation period.

        self.benchmark_mean_return_runs: [float, ]
            List of benchmark mean return for each run.
            It is constant throughout all training process.

        self.mean_r_runs: [[float, ], ]
            List of a list of mean return from 
            optimized model at training period. Each 
            run produces a list of mean return 
            for each optimization step.
        self.mean_constrained_r_runs:  [[float, ], ]
            List of a list of mean return from 
            optimized model at training period using
            constrained weights. Each run produces a 
            list of mean return for each optimization 
            step.
        self.mean_constrained_transaction_r_runs: [[float, ], ]
            List of a list of mean return from 
            optimized model at training period using
            constrained weights and transaction costs. 
            Each run produces a list of mean return for 
            each optimization step.

        self.mean_r_val: [float, ]
            List of validation return using optimized weights through each optimization step.

        self.benchmark_test_r_runs:  [float, ]
            Benchmark mean return for test period 
        self.benchmark_test_r_runs_std: [float, ]
            Benchmark std from return for test period
            at each run.

        self.test_r_runs: [float, ]
            List of mean return on test set for
            each run with optmization model.
        self.test_r_runs_std: [float, ]
            List of std from return on test set for
            each run with optmization model.
        self.test_r_constrained_runs: [float, ]
            List of mean return on test set using
            constrained weights for each run 
            with optmization model.
        self.test_r_constrained_runs_std: [float, ]
            List of std from return on test set using
            constrained weights for each run and
            with optmization model.
        self.test_r_constrained_transaction_runs: [float, ]
            List of mean return on test set using
            constrained weights and transaction costs 
            for each run with optmization model.
        self.test_r_constrained_transaction_runs_std: [float, ]
            List of std from return on test set using
            constrained weights and transaction costs 
            for each run with optmization model.

        self.test_r_nn_runs: [float, ]
            List of mean return on test set for
            each run with neural network model.  
        self.test_r_nn_runs_std: [float, ]
            List of std from return on test set for
            each run with neural network model.  
        self.test_r_nn_constrained_runs: [float, ]
            List of mean return on test set using
            constrained weights for each run 
            with neural network model.
        self.test_r_nn_constrained_runs_std: [float, ]
            List of std from return on test set using
            constrained weights for each run 
            with neural network model.
        self.test_r_nn_constrained_transaction_runs: [float, ]
            List of mean return on test set using
            constrained weights and transaction costs 
            for each run with neural network model.

        self.nn_return_runs: [[float, ], ]
            List of a list of mean return from
            neural network model at traning period.
            Each run produces a list of mean return
            for each epoch.
        self.nn_val_return_runs: [[float, ], ]
            Same as nn_return_runs, only change
            the time period to validation.

        self.nn_return_constrained: [ numpy.Array, ]
            Mean return over epochs using
            constrained weights in training period.
        self.nn_val_return_constrained: [ numpy.Array, ]
            Mean return over epochs using
            constrained weights in validation period.

        self.nn_loss_runs: [[float, ], ]
            List of a list of loss from
            neural network model at traning period.
            Each run produces a list of loss
            for each epoch.
        self.nn_val_loss_runs: [[float, ], ]
            Same as nn_loss_runs, only change
            the time period to validation.

        self.test_cdi_return:
            CDI return sequence at test set.
        self.test_ibov_return:
            IBOV return sequence at test set.

        self.weights_computed: dict( list(numpy.Array, ) )
            Weights computed from NN and OPT best thetas at
            test period. Also save constrained weight versions.

        Returns
        -------
        self.sharp_ratio: numpy.Array
            Array of Sharpe ratios from each
            run at test set with optmized model.
        self.sharp_ratio_constrained : numpy.Array
            Array of Sharpe ratios from each
            run at test set using constrained weights
            with optmized model.
        self.sharp_ratio_nn: numpy.Array
            Array of Sharpe ratios from each
            run at test set with neural network model.
        self.sharp_ratio_constrained_nn: numpy.Array
            Array of Sharpe ratios from each
            run at test set using constrained weights
            with neural network model.

        """
        pathlib.Path('images').mkdir(exist_ok=True)
        
        ## ADD TO DOCUMENTATIO
        stats_df = pd.concat(self.status_df_runs, axis=1).groupby(level=0, axis=1).mean()
        stats_df.to_excel('./sheets/mean_return_statistics.xlsx')

        runs_size = len(self.status_df_runs)
        for run in range(runs_size):
            self.status_df_runs[run].columns += f'Run {run}'

        stats_df = pd.concat(self.status_df_runs, axis=1)
        stats_df.to_excel('./sheets/detailed_return_statistics.xlsx')


        # Train results
        mean_obj_r_runs=self.mean_obj_r_runs
        benchmark_mean_return_runs=self.benchmark_mean_return_runs
        mean_r_runs=self.mean_r_runs
        mean_constrained_r_runs=self.mean_constrained_r_runs
        mean_constrained_transaction_r_runs=self.mean_constrained_transaction_r_runs

        # Test results
        benchmark_test_r_mean = np.mean(self.benchmark_test_r_runs, axis=0)
        benchmark_test_r_mean_std= np.mean(self.benchmark_test_r_runs_std, axis=0)
        cdi_mean_r = np.mean(self.test_cdi_return_runs, axis=0)
        ibov_mean_r = np.mean(self.test_ibov_return_runs, axis=0)

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

        self.sharp_ratio_constrained = np.array(self.test_r_constrained_runs)/np.array(self.test_r_constrained_runs_std)
        self.benchmark_sharp_ratio = np.array(self.benchmark_test_r_runs)/np.array(self.benchmark_test_r_runs_std)
        self.cdi_sharp_ratio = np.array(self.test_cdi_return_runs)/np.array(self.test_cdi_return_std_runs)
        self.ibov_sharp_ratio = np.array(self.test_ibov_return_runs)/np.array(self.test_ibov_return_std_runs)
        self.sharp_ratio_constrained_nn = np.array(self.test_r_nn_constrained_runs)/np.array(self.test_r_nn_constrained_runs_std)

        
        sharp_ratio_constrained = self.sharp_ratio_constrained
        sharp_ratio_constrained_nn = self.sharp_ratio_constrained_nn
        benchmark_sharp_ratio = self.benchmark_sharp_ratio
        ibov_sharp_ratio = self.ibov_sharp_ratio
        cdi_sharp_ratio = self.cdi_sharp_ratio

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
        if self.plot_heatmap == True:
            self.plot_animated_heatmap(self.nn_weight_list, 'nn', experiment_label)
            self.plot_animated_heatmap(self.nn_weight_list_constrained, 'nn_constrained', experiment_label)
        
        # Save best configs for each run
        best_config_df = pd.DataFrame(self.best_config_runs)
        best_config_df.to_csv('./sheets/best_config_throughout_runs.csv')
        
        # Plotting last weights distribution for constrained and unconstrained nn
        last_dist_nn = self.nn_weight_list[-1][-1]
        last_dist_nn_negative = last_dist_nn * (last_dist_nn<0)
        last_dist_nn_positive =last_dist_nn * (last_dist_nn>0)
        last_dist_nn_constrained = self.nn_weight_list_constrained[-1][-1]
        last_dist_nn_constrained_negative = last_dist_nn_constrained * (last_dist_nn_constrained<0)
        last_dist_nn_constrained_positive = last_dist_nn_constrained * (last_dist_nn_constrained>0)

        negative_distance = jensenshannon(last_dist_nn_negative, last_dist_nn_constrained_negative)
        positive_distance = jensenshannon(last_dist_nn_positive, last_dist_nn_constrained_positive)

        LOGGER.info(f"Jensen Shannon distance between positive distributions: {positive_distance}")
        LOGGER.info(f"Jensen Shannon distance between negative distribution: {negative_distance}")

        x = range(len(last_dist_nn))
        plt.figure(figsize=(12,9))
        plt.title("Last NN positive distribution", fontsize=15)
        plt.plot(x, last_dist_nn_positive)
        plt.xlabel('Stock')
        plt.ylabel('Weights')
        plt.grid()
        plt.savefig(f'./images/last_nn_weights_positive_distribution{experiment_label}.jpg')

        x = range(len(last_dist_nn))
        plt.figure(figsize=(12,9))
        plt.title("Last NN negative distribution", fontsize=15)
        plt.plot(x, last_dist_nn_negative)
        plt.xlabel('Stock')
        plt.ylabel('Weights')
        plt.grid()
        plt.savefig(f'./images/last_nn_weights_negative_distribution{experiment_label}.jpg')


        plt.figure(figsize=(12,9))
        plt.title("Last constrained NN positive distribution", fontsize=15)
        plt.plot(x, last_dist_nn_constrained_positive)
        plt.xlabel('Stock')
        plt.ylabel('Weights')
        plt.grid()
        plt.savefig(f'./images/last_constrained_nn_weights_positive_distribution{experiment_label}.jpg')

        plt.figure(figsize=(12,9))
        plt.title("Last constrained NN negative distribution", fontsize=15)
        plt.plot(x, last_dist_nn_constrained_negative)
        plt.xlabel('Stock')
        plt.ylabel('Weights')
        plt.grid()
        plt.savefig(f'./images/last_constrained_nn_weights_negative_distribution{experiment_label}.jpg')

        plt.figure(figsize=(12,9))
        plt.title("Mean utility function for each optimization step", fontsize=15)
        for run, mean_obj_r in enumerate(mean_obj_r_runs):
            x = range(len(mean_obj_r))
            plt.plot(x, mean_obj_r, label=f'Objective return, Run:{run+1}', c=colors[run])
            plt.plot(x, self.mean_obj_r_val_runs[run], label=f'Objective validation return, Run:{run+1}', c=colors[run], linestyle='dashed')
        plt.xlabel('Iteration step')
        plt.ylabel('Objective return')
        plt.legend(fontsize=12)
        plt.grid()
        plt.savefig(f'./images/opt_utility_function_over_steps_{experiment_label}.jpg')
        
        plt.close()

        plt.figure(figsize=(12,9))
        plt.title("Mean return using weight for each optimization step", fontsize=15)
        for run, mean_r in enumerate(mean_r_runs):
            x = range(len(mean_r))
            plt.plot(x, mean_r, label=f'Optimized return, Run:{run+1}', c=colors[run])
            plt.plot(x, [benchmark_mean_return_runs[run]]*len(mean_r), label=f'Benchmark return, Run:{run+1}', c=colors[run], linestyle='dashed')
            plt.plot(x, mean_constrained_r_runs[run], label=f'Optimized return with weight constraints, Run:{run+1}', c=colors[run], linestyle='dotted')
            plt.plot(x, mean_constrained_transaction_r_runs[run], label=f'Optimized return with weight constraints and transaction costs, Run:{run+1}', c=colors[run], linestyle='dashdot')
            plt.plot(x, self.mean_r_val_runs[run], label=f'Optimized validation return, Run:{run+1}', c=colors[run], linestyle='dashed', dash_joinstyle='round', dash_capstyle='round')

            plt.text(x[-1], mean_r[-1]*1.01, f'In sample \nreturn: {mean_r[-1]:.3f}%', fontsize=12)
            plt.text(x[-1], mean_constrained_r_runs[run][-1]*1.01, f'In sample constrained return: \n{mean_constrained_r_runs[run][-1]:.3f}%', fontsize=12)
            plt.text(x[-1], self.mean_r_val_runs[run][-1]*0.95, f'In sample validation \nreturn: {self.mean_r_val_runs[run][-1]:.3f}%', fontsize=12)
        plt.xlabel('Iteration step')
        plt.ylabel('Mean return')
        # plt.legend(loc="lower left")
        # plt.legend()
        plt.grid()
        plt.savefig(f'./images/opt_mean_return_over_steps_{experiment_label}.jpg')
        
        plt.close()
        

        plt.figure(figsize=(12,9))
        plt.title("Train return on NN", fontsize=15)
        for run, mean_r in enumerate(self.nn_return_runs):
            x = range(len(mean_r))
            plt.plot(x, mean_r, label=f'NN return, Run:{run+1}', c=colors[run])
            plt.plot(x, self.nn_val_return_runs[run], label=f'NN validation return, Run:{run+1}', c=colors[run], linestyle='dashed')
            plt.plot(x, self.nn_return_constrained_runs[run], label=f'NN constrained return, Run:{run+1}', c=colors[run], linestyle='dotted')
            plt.plot(x, self.nn_val_return_constrained_runs[run], label=f'NN constrained validation return, Run:{run+1}', c=colors[run], linestyle='dashdot')

            plt.text(x[-1], mean_r[-1]*1.01, f'In sample return: \n{mean_r[-1]:.3f}%', fontsize=12)
            plt.text(x[-1], self.nn_val_return_runs[run][-1]*1.01, f'Validation return: \n{self.nn_val_return_runs[run][-1]:.3f}%', fontsize=12)
            plt.text(x[-1], self.nn_return_constrained_runs[run][-1]*1.01, f'In sample constrained \nreturn: {self.nn_return_constrained_runs[run][-1]:.3f}%', fontsize=12)
            plt.text(x[-1], self.nn_val_return_constrained_runs[run][-1]*1.01, f'Validation constrained \nreturn: {self.nn_val_return_constrained_runs[run][-1]:.3f}%', fontsize=12)
        plt.ylabel("Mean return")
        plt.xlabel("Epochs")
        plt.legend(fontsize=12)
        plt.grid()
        plt.savefig(f'./images/nn_mean_return_over_epochs_{experiment_label}.jpg')

        plt.close()

        plt.figure(figsize=(12,9))
        plt.title("Loss for each epoch", fontsize=15)
        for run, loss in enumerate(self.nn_loss_runs):
            x = range(len(loss))
            plt.plot(x, loss, label=f'Objective loss, Run:{run+1}', c=colors[run])
            plt.plot(x, self.nn_val_loss_runs[run], label=f'Objective validation loss, Run:{run+1}', c=colors[run], linestyle='dashed')
        plt.ylabel("Objective loss")
        plt.xlabel("Epochs")
        plt.legend(fontsize=12)
        plt.grid()
        plt.savefig(f'./images/nn_loss_over_epochs_{experiment_label}.jpg')

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
        plt.close()

        plt.figure(figsize=(12,9))
        plt.title("Mean return on test set", fontsize=15)
        plt.ylabel('Mean return')

        plt.bar(br1, test_r_constrained_mean, label='Optimized test return with weight constraints', color='green', width = 0.09, alpha=0.5)
        plt.bar(br2, test_r_constrained_transaction_mean, label='Optimized test return with weight constraints and transaction costs.', color='black', width = 0.09, alpha=0.5)
        plt.bar(br3, benchmark_test_r_mean, label='Benchmark test return', color='red', width = 0.09, alpha=0.5)
        plt.bar(br4, test_r_nn_constrained, label='Optimized test return using NN with weight constraints', color='coral', width = 0.09, alpha=0.5)
        plt.bar(br5, test_r_nn_constrained_transaction, label='Optimized test return using NN with weight constraints and transaction costs.', color='violet', width = 0.09, alpha=0.5)
        plt.bar(br6, cdi_mean_r, label='CDI/Selic mean return', color='cyan', width = 0.09, alpha=0.5)
        plt.bar(br7, ibov_mean_r, label='IBOV mean return', color='lime', width = 0.09, alpha=0.5)

        plt.text(-0.04, test_r_constrained_mean*1.01, f'Return  :{test_r_constrained_mean:.3f}%', fontsize=12)
        plt.text(width-0.04, test_r_constrained_transaction_mean*1.01, f' Return :{test_r_constrained_transaction_mean:.3f}%', fontsize=12)
        plt.text(2*width-0.04 , benchmark_test_r_mean*1.01, f'Return :{benchmark_test_r_mean:.3f}%', fontsize=12)
        plt.text(3*width-0.04 , test_r_nn_constrained*1.01, f'Return :{test_r_nn_constrained:.3f}%', fontsize=12)
        plt.text(4*width-0.04 , test_r_nn_constrained_transaction*1.01, f'Return :{test_r_nn_constrained_transaction:.3f}%', fontsize=12)
        plt.text(5*width-0.04 , cdi_mean_r*1.01, f'Return :{cdi_mean_r:.3f}%', fontsize=12)
        plt.text(6*width-0.04 , ibov_mean_r*1.01, f'Return :{ibov_mean_r:.3f}%', fontsize=12)
        plt.xticks([])
        plt.legend(loc="center left", fontsize=12)
        plt.savefig(f'./images/mean_return_comparison_test_set_{experiment_label}.jpg')


        plt.figure(figsize=(12,9))
        plt.title("Return on test set for each run", fontsize=15)
        plt.ylabel('Mean return')
        width = 0.1
        br = np.arange(1)
        self.test_r_nn_runs_std
        self.test_r_nn_constrained_runs_std

        for idx, nn_test_return in enumerate(self.test_r_nn_runs):
            inner_br = [x + width*idx for x in br]
            plt.bar(
                inner_br, nn_test_return,
                yerr=self.test_r_nn_runs_std[idx],
                label=f'Test mean return at run: {idx+1}',
                color=colors[idx], width = 0.09, alpha=0.5
            )
            plt.text( width*idx-0.02, nn_test_return*1.01, f'Return  :{nn_test_return:.3f}%', fontsize=12)
        plt.xticks([])
        plt.legend(fontsize=12)
        plt.savefig(f'./images/return_nn_test_set_{experiment_label}.jpg')


        plt.figure(figsize=(12,9))
        plt.title("Return on test set using constrained model for each run", fontsize=15)
        plt.ylabel('Mean return')
        width = 0.1
        br = np.arange(1)
        for idx, constrained_nn_test_return in enumerate(self.test_r_nn_constrained_runs):
            inner_br = [x + width*idx for x in br]
            plt.bar(
                inner_br, constrained_nn_test_return,
                yerr=self.test_r_nn_constrained_runs_std[idx],
                label=f'Constrained test mean return at run: {idx+1}',
                color=colors[idx], width = 0.09, alpha=0.5
            )
            plt.text( width*idx-0.02, constrained_nn_test_return*1.01, f'Return  :{constrained_nn_test_return:.3f}%', fontsize=12)
        plt.xticks([])
        plt.legend(fontsize=12)
        plt.savefig(f'./images/return_nn_constrained_test_set_{experiment_label}.jpg')

        plt.figure(figsize=(12,9))
        plt.title("Constraint mean return comparison")
        plt.ylabel('Mean return')
        plt.bar(br1, test_r_mean, label='Optimized test return',color='blue', width = 0.09, alpha=0.5)
        plt.bar(br2, test_r_constrained_mean, label='Optimized test return with weight constraints', color='green', width = 0.09, alpha=0.5)
        plt.bar(br3, test_r_nn, label='Optimized test return using NN', color='black', width = 0.09, alpha=0.5)
        plt.bar(br4, test_r_nn_constrained, label='Optimized test return using NN with weight constraints', color='red', width = 0.09, alpha=0.5)
        plt.text(-0.04, test_r_mean*1.01, f'Return :{test_r_mean:.3f}%', fontsize=12)
        plt.text(width-0.04, test_r_constrained_mean*1.01, f'Return  :{test_r_constrained_mean:.3f}%', fontsize=12)
        plt.text(2*width-0.04 , test_r_nn*1.01, f'Return :{test_r_nn:.3f}%', fontsize=12)
        plt.text(3*width-0.04 , test_r_nn_constrained*1.01, f'Return :{test_r_nn_constrained:.3f}%', fontsize=12)
        plt.xticks([])
        plt.legend(loc="center left", fontsize=12)
        plt.savefig(f'./images/constraint_mean_return_comparison_test_set_{experiment_label}.jpg')


        if self.plot_weights:
            runs = len(self.weights_computed['optimized'])
            for index in range(runs):
                for weight_type in self.weights_computed:
                    weights = self.weights_computed[weight_type][index]
                    plt.figure(figsize=(12,9))
                    plt.title(f"Stocks {weight_type.replace('_', ' ')} weights over time heatmap", fontsize=15)
                    plt.pcolor(weights.T, cmap=cm.seismic)
                    plt.ylabel("Time")
                    plt.xlabel("Stocks")
                    plt.colorbar().set_label("Weights")
                    plt.savefig(f'./images/stock_{weight_type}_weights_run_{index+1}_{experiment_label}.jpg')
        plt.close()

        plt.figure(figsize=(12,9))

        benchmark_sharp_ratio = np.mean(benchmark_sharp_ratio)
        sharp_ratio_constrained = np.mean(sharp_ratio_constrained)
        ibov_sharp_ratio=np.mean(ibov_sharp_ratio)
        cdi_sharp_ratio=np.mean(cdi_sharp_ratio)
        sharp_ratio_constrained_nn=np.mean(sharp_ratio_constrained_nn)
        plt.title("Mean sharpe ratios", fontsize=15)
        plt.bar(0, benchmark_sharp_ratio, label='Benchmark mean sharpe ratio')
        plt.bar(1, ibov_sharp_ratio, label='Ibov mean sharpe ratio')
        plt.bar(2, sharp_ratio_constrained, label='Optimized constrained mean sharpe ratio')
        plt.bar(3, sharp_ratio_constrained_nn, label='Optimized constrained mean sharpe ratio with NN')
        # plt.bar(4, cdi_sharp_ratio, label='CDI mean sharpe ratio')

        plt.text(-0.3, benchmark_sharp_ratio*1.01, f'Sharpe Ratio: {benchmark_sharp_ratio:.3f}', fontsize=12)
        plt.text(1-0.3, ibov_sharp_ratio*1.01, f'Sharpe Ratio: {ibov_sharp_ratio:.3f}', fontsize=12)
        plt.text(2-0.3, sharp_ratio_constrained*1.01, f'Sharpe Ratio: {sharp_ratio_constrained:.3f}', fontsize=12)
        plt.text(3-0.3, sharp_ratio_constrained_nn*1.01, f'Sharpe Ratio: {sharp_ratio_constrained_nn:.3f}', fontsize=12)
        # plt.text(4-0.3, cdi_sharp_ratio*1.01, f'Sharpe Ratio: {cdi_sharp_ratio:.3f}', fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.savefig(f'./images/mean_sharpe_ratios_{experiment_label}.jpg')

        plt.close()


        plt.figure(figsize=(12,9))
        sharp_ratio = np.array(self.test_r_runs)/np.array(self.test_r_runs_std)
        sharp_ratio_nn = np.array(self.test_r_nn_runs)/np.array(self.test_r_nn_runs_std)
        sharp_ratio = np.mean(sharp_ratio)
        sharp_ratio_constrained = np.mean(sharp_ratio_constrained)
        sharp_ratio_nn=np.mean(sharp_ratio_nn)
        sharp_ratio_constrained_nn=np.mean(sharp_ratio_constrained_nn)
        plt.title("Constrained sharpe ratios comparison", fontsize=15)
        plt.bar(0, sharp_ratio, label='Optimized mean sharpe ratio')
        plt.bar(1, sharp_ratio_constrained, label='Optimized constrained mean sharpe ratio')
        plt.bar(2, sharp_ratio_nn, label='Optimized mean sharpe ratio with NN')
        plt.bar(3, sharp_ratio_constrained_nn, label='Optimized constrained mean sharpe ratio with NN')
        plt.text(-0.3, sharp_ratio*1.01, f'Sharpe Ratio: {sharp_ratio:.3f}', fontsize=12)
        plt.text(1-0.3, sharp_ratio_constrained*1.01, f'Sharpe Ratio: {sharp_ratio_constrained:.3f}', fontsize=12)
        plt.text(2-0.3, sharp_ratio_nn*1.01, f'Sharpe Ratio: {sharp_ratio_nn:.3f}', fontsize=12)
        plt.text(3-0.3, sharp_ratio_constrained_nn*1.01, f'Sharpe Ratio: {sharp_ratio_constrained_nn:.3f}', fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.savefig(f'./images/constrained_sharpe_ratios_{experiment_label}.jpg')


        plt.close()

        ##### LEVERAGE PLOTS ##### -> only using first data to be fast
        weight_types = ['optimized', 'nn_optimized']
        stocks_names = np.array(self.stocks_names)
        for weight_type in weight_types:
            time_range = self.weights_computed[weight_type][0].shape[1]
            w = self.weights_computed[weight_type][0]
            plt.figure(figsize=(12,9))
            plt.title(f"Total Leverage percentage for each test time step using {weight_type.replace('_', ' ')} weights", fontsize=15)
            plt.ylabel("Percentage")
            plt.xlabel("Test time step")
            plt.bar(range(time_range) ,abs(w*(w<0)).sum(axis=0)*100)
            plt.savefig(f'./images/leverage_{weight_type}_{experiment_label}.jpg')

            
            min_leverage_values = abs((w*(w<0)).min(axis=0))*100
            plt.figure(figsize=(12,9))
            plt.title(f"Highest leverage percentage for each test time step using {weight_type.replace('_', ' ')} weights", fontsize=15)
            plt.ylabel("Percentage")
            plt.xlabel("Test time step")
            plt.bar(range(time_range) , min_leverage_values)
            for i in range(time_range):
                plt.text(i-0.4, min_leverage_values[i]*1.01, stocks_names[i], fontsize=8)
            plt.savefig(f'./images/highest_leverage_{weight_type}_{experiment_label}.jpg')
        plt.close()
        

        #### COMPARISON WITH IBOV AND SELIC/CDI ######
        comparison_fields = ["cdi", "ibov"]
        calculated_fields = ["nn_constraint", "opt_constraint"]
        
        for comp_field in comparison_fields:
            for cal_field in calculated_fields:
                plt.figure(figsize=(12,9))
                plt.title(f"Comparisson between {cal_field} and {comp_field} results.", fontsize=15)
                y = self.results_comparison[f"test_{cal_field}_{comp_field}_comparison"]
                x = range(len(y))
                plt.ylabel("Difference between return")
                plt.xlabel("Month on test set")
                plt.bar(x, y)
                plt.grid()
                plt.savefig(f'./images/comparison_{cal_field}_{comp_field}_{experiment_label}.jpg')
        
        LOGGER.info("Saved plots in folder.")


    def _start(self):
        """
        Start portfolio optimization.
        """
        self.load_data()

        self.stocks_names = list(self.monthly_return.columns)
        total_size = self.monthly_return.shape[0]

        indexes_list = []
        for train_percentage, val_percentage, test_percentage in zip(self.train_split, self.val_split, self.test_split):
            idxs_list, = data_split(total_size, train_percentage, val_percentage, test_percentage)
            indexes_list.append(idxs_list)
        pathlib.Path("sheets").mkdir(exist_ok=True)
        self.create_experiment(indexes_list)

        experiment_label = "testing_comparison"
        self.plot_final_results(experiment_label)
        LOGGER.info("Done")

def main():
    data_path = "../data/"
    risk_constant = 5
    np.random.seed(123)
    train_split = [0.2, 0.4, 0.6]
    val_split = [0.2, 0.2, 0.2]
    test_split = [0.2, 0.2, 0.2]

    single_holdout = ParametricPortifolio(
        data_path=data_path, risk_constant=risk_constant,
        train_split=train_split, val_split=val_split,
        test_split=test_split, benchmark_type='value_weighted',
        plot_weights=False, plot_heatmap=False,
        )
    single_holdout._start()


if __name__ == "__main__":
    main()