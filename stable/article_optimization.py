import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from collections import defaultdict
from scipy.optimize import minimize
import time as tm
import os
import logging


from utils import (
    data_split, create_w_benchmark, plot_splitted_data,
    compute_transaction_discount, utility_function,
    constrain_weights, 
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
format_string = '%(asctime)s [%(process)d.%(thread)d] %(levelname)-3s %(name)-3s  %(funcName)s(%(lineno)d): %(message)s'
formatter = logging.Formatter(format_string)
handler.setFormatter(formatter)
LOGGER.addHandler(handler)


class ParametricPortifolio():
    """
    """

    def __init__(self, data_path, risk_constant, transaction_cost, train_split):
        """
        """
        self.data_path = data_path
        self.risk_constant = risk_constant
        self.transaction_cost = transaction_cost
        self.train_split = train_split

        self.mean_obj_r_runs = []
        self.mean_r_runs = []
        self.benchmark_mean_return_runs = []
        self.benchmark_test_r_runs = []
        self.benchmark_test_r_runs_std = []
        self.test_r_runs = []
        self.test_r_runs_std = []

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
        and dividing it by the sum of all stocks at each time. Bringing it to values between 0 and 1.

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


        epsilon = 1e-10
        for name in characteristics_names:
            #Normalize firm characteristics for all stocks
            sum_df = firm_characteristics.T.loc[(slice(None), name), :].sum()
            firm_characteristics.T.loc[(slice(None), name), :] -= firm_characteristics.T.loc[(slice(None), name), :].mean()
            firm_characteristics.T.loc[(slice(None), name), :] /= (sum_df + epsilon)
        LOGGER.info("Normalized firm characteristics")
        return firm_characteristics
    

    def optimizing_step(self, firm_characteristics, r, time, number_of_stocks, theta0, w_benchmark):
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
        Returns
        -------
        sol: scipy.optimize.OptimizeResult
            Solution object used to retrieve theta optimized and optimization information.
        mean_obj_r: [float, ]
            List of objective values through each optimization step.
        mean_r: [float, ]
            List of return using optimized weights through each optimization step.
        """
        LOGGER.info("Started optimization step.")

        # Get constants
        risk_constant = self.risk_constant
        transaction_cost = self.transaction_cost

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
            return -sum(sum(utility_function(risk_constant, w[:,:-1]*r[:,1:])))/time
        
        mean_obj_r = []
        mean_r = []
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
            for i in range(number_of_stocks):
                w_iter[i] = w_benchmark[i] + (1/number_of_stocks)*thetaI.dot(firm_characteristics[i].copy().T)
            w_iter_constrained = constrain_weights(w_iter.copy())
            transaction_discount = compute_transaction_discount(w_iter_constrained, transaction_cost)

            mean_r.append(sum(sum(w_iter*r))/time)
            mean_constrained_r.append(sum(sum(w_iter_constrained*r))/time)
            mean_constrained_transaction_r.append(sum(sum(w_iter_constrained*r))/time  - transaction_discount)

        sol = minimize(objective, theta0, callback=callback_steps, method='BFGS')
        self.sol = sol
        self.mean_obj_r  = mean_obj_r
        self.mean_r = mean_r
        self.mean_constrained_r = mean_constrained_r
        self.mean_constrained_transaction_r = mean_constrained_transaction_r
        LOGGER.info("Finished optimization step.")
    
    # TO-DO: Change it to load any number of characteristics
    # BUY-HOLD STRATEGY
    def evaluate_theta(self, sol_theta, test_me, test_mom, test_btm, test_return):
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
        stocks_names: [str, ]
            List of stocks to be looked up.

        Returns
        -------

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
        transaction_cost = self.transaction_cost
        #### TESTING CHARACTERISTICS
        firm_characteristics_test, r_test, time_test, number_of_stocks = self.create_characteristics(test_me, test_mom, test_btm, test_return)

        #### CREATE BENCHMARK FOR TESTING
        w_benchmark_test = create_w_benchmark(number_of_stocks, time_test)

        benchmark_r_test_series=pd.Series(sum(w_benchmark_test*r_test)).describe()
        benchmark_test_r = benchmark_r_test_series['mean']
        benchmark_test_r_std = benchmark_r_test_series['std']

        ### CREATE TEST WEIGHT AND FIND ITS RETURN
        w_test = np.empty(shape=(number_of_stocks, time_test))
        for i in range(number_of_stocks):
            firm_df = firm_characteristics_test[i].copy()
            firms_coeff = sol_theta.dot(firm_df.T)
            w_test[i] = w_benchmark_test[i] + (1/number_of_stocks)*firms_coeff
        
        w_test_constrained = constrain_weights(w_test.copy())
        
        transaction_discount = compute_transaction_discount(w_test_constrained, transaction_cost)

        r_test_series = pd.Series(sum(w_test*r_test)).describe()
        r_test_constrained_series = pd.Series(sum(w_test_constrained*r_test)).describe()

        test_r = r_test_series['mean']
        test_r_std = r_test_series['std']

        test_r_constrained = r_test_constrained_series['mean']
        test_r_std_constrained_std = r_test_constrained_series['std']

        test_r_constrained_transaction = test_r_constrained-transaction_discount

        # import pdb; pdb.set_trace()

        self.benchmark_test_r = benchmark_test_r
        self.benchmark_test_r_std  = benchmark_test_r_std
        self.test_r = test_r
        self.test_r_std = test_r_std
        self.test_r_constrained = test_r_constrained
        self.test_r_std_constrained_std = test_r_std_constrained_std
        self.test_r_constrained_transaction = test_r_constrained_transaction

        LOGGER.info("Evalueted theta on test set.")
        # return benchmark_test_r, benchmark_test_r_std, test_r, test_r_std, test_r_constrained, test_r_std_constrained_std, test_r_constrained_transaction
    
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
        stocks_names = self.stocks_names
        np.random.seed(123)
        for train, test in indexes_list:
            LOGGER.info("Splitting data into train and test set.")
            theta0 = np.random.rand(1, 3)
            
            train_btm = self.book_to_mkt_ratio.loc[train]
            train_me = self.mcap.loc[train]
            train_mom = self.lreturn.loc[train]
            train_return = self.monthly_return.loc[train]
            
            test_btm = self.book_to_mkt_ratio.loc[test]
            test_me = self.mcap.loc[test]
            test_mom = self.lreturn.loc[test]
            test_return = self.monthly_return.loc[test]

            #### TRAINING CHARACTERISTICS
            firm_characteristics, r, time, number_of_stocks = self.create_characteristics(train_me, train_mom, train_btm, train_return)

            ### Creating weights to a benchmark portifolio using uniform weighted returns
            w_benchmark = create_w_benchmark(number_of_stocks, time)

            ### CREATING RETURNS TO COMPARE
            benchmark_mean_return = sum(sum(w_benchmark*r))/time
            
            # Creating optimizing solution sol
            self.optimizing_step(firm_characteristics, r, time, number_of_stocks, theta0, w_benchmark)

            sol_theta = self.sol.x

            ### Evaluate founded theta on test samples and find its mean return from optimized and benchmark.
            self.evaluate_theta(sol_theta, test_me, test_mom, test_btm, test_return)
            
            self.mean_obj_r_runs.append(self.mean_obj_r)
            self.mean_r_runs.append(self.mean_r)
            self.benchmark_mean_return_runs.append(benchmark_mean_return)
            self.benchmark_test_r_runs.append(self.benchmark_test_r)
            self.benchmark_test_r_runs_std.append(self.benchmark_test_r_std)
            self.test_r_runs.append(self.test_r)
            self.test_r_runs_std.append(self.test_r_std)

            LOGGER.info("Finished experiment.")
    

    def plot_final_results(self, experiment_label):
        """
        Plot returns through optimization and in benchmark test.
        """
        
        mean_obj_r_runs=self.mean_obj_r_runs
        mean_r_runs=self.mean_r_runs
        benchmark_mean_return_runs=self.benchmark_mean_return_runs
        test_r_runs=self.test_r_runs
        benchmark_test_r_runs = self.benchmark_test_r_runs
        test_r_runs_std=self.test_r_runs_std
        benchmark_test_r_runs_std=self.benchmark_test_r_runs_std
        mean_constrained_r=self.mean_constrained_r
        mean_constrained_transaction_r=self.mean_constrained_transaction_r
        test_r_constrained=self.test_r_constrained
        test_r_std_constrained_std=self.test_r_std_constrained_std
        test_r_constrained_transaction=self.test_r_constrained_transaction
        

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

        plt.figure(figsize=(12,9))
        plt.title("Mean objective return for each optimization step")
        for run, mean_obj_r in enumerate(mean_obj_r_runs):
            x = range(len(mean_obj_r))
            plt.plot(x, mean_obj_r, label=f'Objective return, Run:{run+1}', c=colors[run])
        plt.xlabel('Iteration step')
        plt.ylabel('Objective return')
        plt.legend()
        plt.grid()
        plt.savefig(f'./{experiment_label}_objective_return_over_steps.jpg')
        # plt.show()

        plt.figure(figsize=(12,9))
        plt.title("Mean return using weight for each optimization step")
        for run, mean_r in enumerate(mean_r_runs):
            x = range(len(mean_r))
            plt.plot(x, mean_r, label=f'Optimized return, Run:{run+1}', c=colors[run])
            plt.plot(x, [benchmark_mean_return_runs[run]]*len(mean_r), label=f'Benchmark return, Run:{run+1}', c=colors[run], linestyle='dashed')
            plt.plot(x, mean_constrained_r, label=f'Optimized return with weight constrain, Run:{run+1}', c=colors[run+1])
            plt.plot(x, mean_constrained_transaction_r, label=f'Optimized return with weight constrain and transaction costs, Run:{run+1}', c=colors[run+2])
        plt.xlabel('Iteration step')
        plt.ylabel('Mean return')
        plt.legend()
        plt.grid()
        plt.savefig(f'./{experiment_label}_mean_return_over_steps.jpg')
        # plt.show()


        width = 0.1
        br1 = np.arange(len(test_r_runs))
        br2 = [x + width for x in br1]
        br3 = [x + width for x in br2]
        plt.figure(figsize=(12,9))
        plt.title("Mean return on test set with standard deviation.")
        plt.ylabel('Mean return')
        plt.bar(br1, test_r_runs, label='Optimized test return', yerr=test_r_runs_std, color='blue', width = 0.09)
        plt.bar(br2, test_r_constrained, label='Optimized test return with weight constrain', yerr= test_r_std_constrained_std, color='green', width = 0.09)
        plt.bar(br3, benchmark_test_r_runs, label='Benchmark test return', yerr=benchmark_test_r_runs_std, color='red', width = 0.09)
        plt.xticks([])
        plt.grid()
        plt.legend()
        plt.savefig(f'./{experiment_label}_mean_return_test_set_with_std.jpg')
        # plt.show()

        plt.figure(figsize=(12,9))
        plt.title("Mean return on test set")
        plt.ylabel('Mean return')
        br4 = [x + width for x in br3]
        plt.bar(br1, test_r_runs, label='Optimized test return',color='blue', width = 0.09, alpha=0.5)
        # plt.axhline(test_r_runs,color='blue', label='Optimized test return mean.')

        plt.bar(br2, test_r_constrained, label='Optimized test return with weight constrain', color='green', width = 0.09, alpha=0.5)
        # plt.axhline(test_r_constrained,color='green', label='Optimized test return with weight constrain.')

        plt.bar(br3, test_r_constrained_transaction, label='Optimized test return with weight constrain and transaction costs.', color='black', width = 0.09, alpha=0.5)
        # plt.axhline(test_r_constrained_transaction,color='black', label='Optimized test return mean with weight constrain and transaction costs.')

        plt.bar(br4, benchmark_test_r_runs, label='Benchmark test return', color='red', width = 0.09, alpha=0.5)
        # plt.axhline(benchmark_test_r_runs,color='red', label='Benchmark test return mean.')


        plt.text(-0.03, test_r_runs[0]*1.01, f'Return :{test_r_runs[0]:.3f}%')
        plt.text(width-0.03, test_r_constrained*1.01, f'Return  :{test_r_constrained:.3f}%')
        plt.text(2*width-0.03, test_r_constrained_transaction*1.01, f' Return :{test_r_constrained_transaction:.3f}%')
        plt.text(3*width-0.03 , benchmark_test_r_runs[0]*1.01, f'Return :{benchmark_test_r_runs[0]:.3f}%')

        plt.xticks([])
        plt.legend(loc="lower right")
        plt.savefig(f'./{experiment_label}_mean_return_test_set.jpg')
        LOGGER.info("Saved plots in folder.")

    def _start(self):
        self.load_data()

        self.stocks_names = list(self.monthly_return.columns)
        total_size = self.monthly_return.shape[0]

        indexes_list = data_split(total_size, self.train_split)
        
        self.create_experiment(indexes_list)
        
        experiment_label = "OO_experiment_holdout"
        self.plot_final_results(experiment_label)

        LOGGER.info("Done")

def main():
    data_path = "../data/"
    transaction_cost=0.03
    risk_constant = 5
    train_split = 0.7

    single_holdout = ParametricPortifolio(
        data_path=data_path, transaction_cost=transaction_cost, risk_constant=risk_constant,
        train_split=train_split
        )
    single_holdout._start()


if __name__ == "__main__":
    main()