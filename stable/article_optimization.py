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

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
format_string = '%(asctime)s [%(process)d.%(thread)d] %(levelname)-3s %(name)-3s  %(funcName)s(%(lineno)d): %(message)s'
formatter = logging.Formatter(format_string)
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

# TO-DO: Change it to load any number of characteristics
def load_data(data_path):
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
    lreturn = pd.read_csv(os.path.join(data_path, "monthly_lagged_return.csv"))
    mcap = pd.read_csv(os.path.join(data_path, "monthly_market_cap.csv"))
    book_to_mkt_ratio = pd.read_csv(os.path.join(data_path, "monthly_book_to_mkt_ratio.csv"))
    monthly_return = pd.read_csv(os.path.join(data_path, "monthly_return.csv"))
    lreturn.fillna(method='bfill', inplace=True)
    lreturn.fillna(method='ffill', inplace=True)
    mcap.fillna(method='bfill', inplace=True)
    mcap.fillna(method='ffill', inplace=True)
    book_to_mkt_ratio.fillna(method='bfill', inplace=True)
    book_to_mkt_ratio.fillna(method='ffill', inplace=True)
    monthly_return.fillna(method='bfill', inplace=True)
    monthly_return.fillna(method='ffill', inplace=True)

    LOGGER.info("Loaded data successfully")
    return mcap, lreturn, book_to_mkt_ratio, monthly_return



def normalize_characteristics(firm_characteristics, characteristics_names):
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

# TO-DO: Change it to create any number of characteristics
def create_characteristics(me_df, mom_df, btm_df, return_df, stocks_names):
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
    firm_characteristics = normalize_characteristics(firm_characteristics, characteristics_names)

    return firm_characteristics, r, time, number_of_stocks

def utility_function(risk_factor, portfolio_return):
    """
    CRRA Utility function

    Parameters
    ----------
    risk_factor: int
        Risk constant, increase it to become more risk averse.
    portifolio_return: float
        Mean return of a portifolio.

    Returns
    -------
    value: float
        Utility function value
    """
    value = ((1 + portfolio_return)**(1-risk_factor))/(1-risk_factor)
    return value

def optimizing_step(firm_characteristics, r, time, number_of_stocks, theta0, w_benchmark, risk_constant):
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
        mean_r.append(sum(sum(w_iter*r))/time)

    sol = minimize(objective, theta0, callback=callback_steps, method='BFGS')
    LOGGER.info("Finished optimization step.")
    return sol, mean_obj_r, mean_r

def create_w_benchmark(number_of_stocks, time):
    """
    Create benchmark weights, uniform weighted on the number of stocks .

    Parameters
    ----------
    number_of_stocks: int
        Number of stocks to be analized.
    time: int
        Number of periods of time this slice has.
    Returns
    -------
    w_bechmark: numpy.array
        Benchmark weights with shape equal to (number_of_stocks, time).
    """
    w_benchmark = np.ones(shape=(number_of_stocks, time))
    w_benchmark *= 1/number_of_stocks
    LOGGER.info("Created benchmark weights")
    return w_benchmark

def data_split(size, train_percentage):
    """
    Define indexes for splitted train and test set using a split percentage.

    Parameters
    ----------
    size: int
        Total size of the data to be splitted.
    train_percentage: float
        Percentage between 0 and 1, where the training and testing data are splitted.
    
    Returns
    -------
    indexes_list: [( [int, ], [int, ] )]
        List of tuples containing training indexes and testing indexes.

    """
    train_size = int(size*train_percentage)
    train_index =np.arange(train_size) 
    test_index = np.arange(start=train_size, stop=size)
    indexes_list = [(train_index, test_index)]
    LOGGER.info(f'Train size: {train_size}, test size: {size-train_size}')
    return indexes_list

def plot_splitted_data(characteristic_df, characteristic_name, indexes_list, stock_name):
    """
    Plot train and test splitted data and save it to a jpg file in the folder.
    
    Parameters
    ----------
    characteristic_df : pandas.DataFrame
        A characteristic dataframe to be splitted.
    characteristic_name: str
        The name of the characteristic used here.
    indexes_list: [( [int, ], [int, ] )]
        List of tuples containing training indexes and testing indexes.
    stock_name: str
        The stock tick to be used for lookup.
    """
    LOGGER.info("PLotting splitted data example.")
    plt.figure(figsize=(12,8))
    plt.title(f"{stock_name} Monthly {characteristic_name}")
    plt.ticklabel_format(style='plain')
    for train, test in indexes_list:
        train_btm = characteristic_df.loc[train]
        test_btm = characteristic_df.loc[test]
        
        train_btm[stock_name].plot(c='blue', label='train')
        test_btm[stock_name].plot(c='orange', label='test')
        plt.xlabel('Month')
        plt.ylabel(characteristic_name)
        plt.legend()
    plt.savefig(f'./{stock_name}_monthly_{characteristic_name.lower()}.jpg')
    # plt.show()


# TO-DO: Change it to load any number of characteristics
def evaluate_theta(sol_theta, test_me, test_mom, test_btm, test_return, stocks_names):
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
    #### TESTING CHARACTERISTICS
    firm_characteristics_test, r_test, time_test, number_of_stocks = create_characteristics(test_me, test_mom, test_btm, test_return, stocks_names)

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
    
    r_test_series = pd.Series(sum(w_test*r_test)).describe()
    test_r = r_test_series['mean']
    test_r_std = r_test_series['std']

    LOGGER.info("Evalueted theta on test set.")
    # import pdb; pdb.set_trace()
    return benchmark_test_r, benchmark_test_r_std, test_r, test_r_std


# TO-DO: Change it to use any number of characteristics
def create_experiment(mcap, lreturn, book_to_mkt_ratio, monthly_return , stocks_names, indexes_list, risk_constant):
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
    for train, test in indexes_list:
        LOGGER.info("Splitting data into train and test set.")
        theta0 = np.random.rand(1, 3)
        
        train_btm = book_to_mkt_ratio.loc[train]
        train_me = mcap.loc[train]
        train_mom = lreturn.loc[train]
        train_return = monthly_return.loc[train]
        
        test_btm = book_to_mkt_ratio.loc[test]
        test_me = mcap.loc[test]
        test_mom = lreturn.loc[test]
        test_return = monthly_return.loc[test]

        #### TRAINING CHARACTERISTICS
        firm_characteristics, r, time, number_of_stocks = create_characteristics(train_me, train_mom, train_btm, train_return, stocks_names)

        ### Creating weights to a benchmark portifolio using uniform weighted returns
        w_benchmark = create_w_benchmark(number_of_stocks, time)

        ### CREATING RETURNS TO COMPARE
        benchmark_mean_return = sum(sum(w_benchmark*r))/time

        
        sol, mean_obj_r, mean_r = optimizing_step(firm_characteristics, r, time, number_of_stocks, theta0, w_benchmark, risk_constant)
        sol_theta = sol.x

        ### Evaluate founded theta on test samples and find its mean return from optimized and benchmark.
        benchmark_test_r, benchmark_test_r_std, test_r, test_r_std = evaluate_theta(sol_theta, test_me, test_mom, test_btm, test_return, stocks_names)

        LOGGER.info("Finished experiment.")
        return benchmark_mean_return, mean_obj_r, mean_r, benchmark_test_r, benchmark_test_r_std, test_r, test_r_std


def plot_final_results(
    mean_obj_r_runs, mean_r_runs, benchmark_mean_return_runs, test_r_runs, 
    benchmark_test_r_runs, test_r_runs_std, benchmark_test_r_runs_std
    ):
    """

    """

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
    plt.savefig(f'./objective_return_over_steps.jpg')
    # plt.show()

    plt.figure(figsize=(12,9))
    plt.title("Mean return using weight for each optimization step")
    for run, mean_r in enumerate(mean_r_runs):
        x = range(len(mean_r))
        plt.plot(x, mean_r, label=f'Optimized return, Run:{run+1}', c=colors[run])
        plt.plot(x, [benchmark_mean_return_runs[run]]*len(mean_r), label=f'Benchmark return, Run:{run+1}', c=colors[run], linestyle='dashed')
    plt.xlabel('Iteration step')
    plt.ylabel('Mean return')
    plt.legend()
    plt.grid()
    plt.savefig(f'./mean_return_over_steps.jpg')
    # plt.show()


    width = 0.1
    br1 = np.arange(len(test_r_runs))
    br2 = [x + width for x in br1]
    plt.figure(figsize=(12,9))
    plt.title("Mean return on test set")
    plt.ylabel('Mean return')
    plt.bar(br1, test_r_runs, label='Optimized test return', yerr=test_r_runs_std, color='blue', width = 0.09)
    plt.bar(br2, benchmark_test_r_runs, label='Benchmark test return', yerr= benchmark_test_r_runs_std, color='red', width = 0.09)
    plt.xticks([])
    plt.grid()
    plt.legend()
    plt.savefig(f'./mean_return_test_set.jpg')
    # plt.show()
    LOGGER.info("Saved plots in folder.")


def main():
    data_path = "../data/"
    mcap, lreturn, book_to_mkt_ratio, monthly_return = load_data(data_path)
    stocks_names = list(monthly_return.columns)
    total_size = monthly_return.shape[0]
    train_split = 0.7
    indexes_list = data_split(total_size, train_split)
    # plot_splitted_data(monthly_return, 'Return', indexes_list, 'VALE3')
    risk_constant = 5
    benchmark_mean_return, mean_obj_r, mean_r, benchmark_test_r, benchmark_test_r_std, test_r, test_r_std = create_experiment(
        mcap, lreturn, book_to_mkt_ratio, monthly_return , stocks_names, indexes_list, risk_constant
    )

    mean_obj_r_runs = [mean_obj_r]
    mean_r_runs = [mean_r]
    benchmark_mean_return_runs = [benchmark_mean_return]
    benchmark_test_r_runs = [benchmark_test_r]
    benchmark_test_r_runs_std = [benchmark_test_r_std]
    test_r_runs = [test_r]
    test_r_runs_std = [test_r_std]
    
    plot_final_results(
        mean_obj_r_runs, mean_r_runs, benchmark_mean_return_runs, test_r_runs, 
        benchmark_test_r_runs , test_r_runs_std, benchmark_test_r_runs_std
        )

    print("Done")


if __name__ == "__main__":
    main()
