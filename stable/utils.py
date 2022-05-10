"""
"""
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from scipy.stats import skew
import torch.nn.functional as F
torch.random.manual_seed(123)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
format_string = '%(asctime)s [%(process)d.%(thread)d] %(levelname)-3s %(name)-3s  %(funcName)s(%(lineno)d): %(message)s'
formatter = logging.Formatter(format_string)
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

def constrain_weights(w):
    """
    Constrain weights to have only the permitted leverage

    Parameters
    ----------

    w: numpy.Array
        Array of weights computed by its models.
        May or not have leverage larger than permited_leverage.

    Returns
    -------

    new_w: numpy.Array
        New weights with only permited leverage
        at each time step
    """

    # Hardcoded
    permited_leverage = -0.3
    must_have_positive_weights = 1.3

    total_negative = (w*(w<0)).sum(axis=0) 
    total_positive = (w*(w>0)).sum(axis=0)

    # If total negative or positive of each time doesn't have any samples we will
    # have infinity values, so to avoid this we set them to 0
    capped_negative_weights = ((w/total_negative)*permited_leverage)
    capped_negative_weights[capped_negative_weights==np.inf] = 0
    capped_negative_weights[capped_negative_weights==-np.inf] = 0
    
    capped_positive_weights = ((w/total_positive)*must_have_positive_weights)
    capped_positive_weights[capped_positive_weights==np.inf] = 0
    capped_positive_weights[capped_positive_weights==-np.inf] = 0

    negative_mask = w<0
    positive_mask = w>0
    new_w = capped_negative_weights*negative_mask+capped_positive_weights*positive_mask
    return new_w

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


def data_split(size, train_percentage, val_percentage, test_percentage):
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
    indexes_list: [( [int, ], [int, ] [int, ])]
        List of tuples containing training indexes and testing indexes.

    """
    train_size = int(size*train_percentage)
    val_end_split = int(size*(train_percentage+val_percentage))
    test_end_split = int(size*(train_percentage+val_percentage+test_percentage))
    train_index =np.arange(train_size)
    val_index = np.arange(start=train_size, stop=val_end_split)
    if test_end_split < size:
        test_index = np.arange(start=val_end_split, stop=test_end_split)
    else:
        test_index = np.arange(start=val_end_split, stop=size)
    indexes_list = [(train_index, val_index, test_index)]
    LOGGER.info(f'Train size: {train_size}, val size: {val_end_split-train_size}, test size: {test_end_split-val_end_split}')
    return indexes_list


def compute_transaction_discount(weights, transaction_cost):
    # Check first stocks bought
    stocks_bought = (weights[0,:]>0).sum()
    row_size = weights.shape[0]
    for i in range(0, row_size-1):
        sliced_array = weights[i:i+2,:].copy()
        # if increase a position compute it as a buy operation
        new_positions = (sliced_array[0] - sliced_array[1] < 0).sum()
        stocks_bought += new_positions

    # Normalize it to have the discount for return at each time
    transaction_discount = stocks_bought*transaction_cost/weights.size
    return transaction_discount

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

# TO-DO: handle multiple manners to constrain weights
def constrain_weights_hard(weights):
    """
    Constrain weights to be nonnegative.

    Parameters
    ----------
    weights: numpy.Array
        Weights after optimization with short positions (negative weights).

    Returns
    -------
    weights: numpy.Array
        Weights processed and normalized to sum 1 each time.
    """
    weights = np.where(weights>0, weights, 0)
    weights = weights/weights.sum(axis=0)
    return weights

def compute_transaction_costs(data_path):
    """
    Compute transaction costs using volume values
    and putting in a comission tax.

    Parameters
    ----------
    data_path: str
        Path where all data is to be found.

    Returns
    -------
    market_cost: numpy.Array
        Array with market cost using
        stock volume as base.
    """

    volumes = pd.read_csv(os.path.join(data_path , 'monthly_volume.csv'))
    comission = 0.0004
    min_tax = 0.002
    max_tax = 0.01

    market_cost = np.ones_like(volumes)*comission
    market_cost.shape

    x= volumes/volumes.std()
    (T,N) = x.shape
    x[x==0]= np.NaN
    min_ = x.min(axis=1)
    max_ = x.max(axis=1)
    for t in range(T):
        x.iloc[t] = (x.iloc[t] - min_[t])/(max_[t] - min_[t])
    x.fillna(0, inplace=True)
    x = x * (min_tax - max_tax)

    liq_cost = np.ones_like(x)*max_tax + x
    market_cost += liq_cost
    market_cost = 1-market_cost
    market_cost = market_cost
    return market_cost


class ParametricPortifolioNN(nn.Module):
    def __init__(self, benchmark, return_, risk_constant, number_of_stocks):
        super(ParametricPortifolioNN, self).__init__()
        # self.first_layer = nn.Linear(3,3)
        # self.second_layer = nn.Linear(3,3)
        self.weights = nn.Linear(3,1, bias=False)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.benchmark = benchmark
        self.return_ = return_
        self.risk_constant = risk_constant
        self.number_of_stocks = number_of_stocks

    def utility_function(self, portfolio_return):
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
        value = ((1 + portfolio_return)**(1-self.risk_constant))/(1-self.risk_constant)
        return value

    def forward(self, x):
        """
        x -> Shape (T, N, 3)
        """
        # x = F.relu(self.first_layer(x))
        # x = F.relu(self.second_layer(x))
        x = self.lrelu(self.weights(x))
        x = x.squeeze(-1) * (1/self.number_of_stocks)
        x = x + self.benchmark
        r = x*self.return_
        x = torch.mean(r, -1)
        x = self.utility_function(x)
        x = torch.mean(x, 0)

        return x , r

def loss_fn(x):
    """ Simple loss function repeating foward calculations """
    return -x


def convert_to_nn_variables(firm_characteristics, r, w_benchmark):
    """
    Convert numpy variables to torch variables
    
    Parameters
    ----------
    firm_characteristics: pd.DataFrame(float)
        Normalized firm_characteristics
        
    r: np.Array(float)
        Monthly return
    
    w_benchmark: np.Array(float)
        Benchmark weights
    
    Returns
    -------
    torch_characteristics: torch.Tensor
        Torch mapped firm characteristics
    torch_r: torch.Tensor
        Torch mapped Monthly return
    torch_benchmark torch.Tensor
        Torch mapped Benchmark weights
    """
    
    nd_characteristics = np.array([
    firm_characteristics.T.loc[(slice(None), 'me'), :],
    firm_characteristics.T.loc[(slice(None), 'btm'), :],
    firm_characteristics.T.loc[(slice(None), 'mom'), :],
    ]).transpose(2,1,0)
    
    torch_characteristics = torch.tensor(nd_characteristics).float()
    torch_r = torch.tensor(r.T)
    
    torch_benchmark = torch.tensor(w_benchmark.T)
    
    return torch_characteristics, torch_r, torch_benchmark


def weight_reset(m):
    """
    Reset neural network weights to retrain it
    """
    if isinstance(m, nn.Linear):
        m.reset_parameters()

## Need to change to more characteristics
def calculate_best_validation_technique(firm_characteristics):
    """
    Calculate best validation technique based on each 
    time series from firm characteristics.

    Parameters
    ----------
    firm_characteristics: pandas.DataFrame
        Dataframe containing all characteristics for 
        all stocks at each time step.

    """
    btm_list = []
    me_list = []
    mom_list = []

    btm_q_list = []
    me_q_list = []
    mom_q_list = []

    btm_95_list = []
    me_95_list = []
    mom_95_list = []

    btm_s_list = []
    me_s_list = []
    mom_s_list = []

    btm_features = defaultdict(list)
    me_features = defaultdict(list)
    mom_features = defaultdict(list)

    number_of_characteristics = 3
    
    LOGGER.info("Started to calculate the best validation technique.")

    number_of_stocks = firm_characteristics.shape[1] // number_of_characteristics
    sm_char = firm_characteristics.rolling(12, min_periods=1).mean()
    em_char = firm_characteristics.ewm(alpha=0.1, adjust=False).mean()
    q5 = firm_characteristics.quantile(0.05)
    q95 = firm_characteristics.quantile(0.95)

    for i in range(number_of_stocks):        
        # Acceleration calculation
        btm_ = sm_char[i]['btm'].mean() / em_char[i]['btm'].mean()
        me_ = sm_char[i]['me'].mean() / em_char[i]['me'].mean()
        mom_ = sm_char[i]['mom'].mean() / em_char[i]['mom'].mean()
    
        btm_s = skew(firm_characteristics[i]['btm'])
        me_s = skew(firm_characteristics[i]['me'])
        mom_s = skew(firm_characteristics[i]['mom'])

        btm_features['q5'].append(q5[i]['btm'])
        btm_features['q95'].append(q95[i]['btm'])
        btm_features['skew'].append(btm_s)
        btm_features['accel'].append(btm_)

        me_features['q5'].append(q5[i]['me'])
        me_features['q95'].append(q95[i]['me'])
        me_features['skew'].append(me_s)
        me_features['accel'].append(me_)

        mom_features['q5'].append(q5[i]['mom'])
        mom_features['q95'].append(q95[i]['mom'])
        mom_features['skew'].append(mom_s)
        mom_features['accel'].append(mom_)
    
    total_passed_tests = 0

    # Stocks with acceleration < 1.2
    total_passed_tests += (np.array(btm_features['acceç']) < 1.2).sum()
    total_passed_tests += (np.array(me_features['accel']) < 1.2).sum()
    total_passed_tests += (np.array(mom_features['accel']) < 1.2).sum()
    
    threshold = (number_of_stocks*number_of_characteristics)//2
    
    # If half of the possible tests not passed
    if total_passed_tests < threshold:
        LOGGER.info(
            f"Only {total_passed_tests} tests passed " + 
            f"out of {number_of_stocks*number_of_characteristics}."
            )
        LOGGER.info("You shoud use REP-Holdout validation instead of Holdout.")
        LOGGER.info("You may stop this experiment and choose it.")
        LOGGER.info("Finished calculation of best validation technique.")
        return
    
    total_passed_tests = 0

    # Stocks with Percentil05 < -1.6
    total_passed_tests += (np.array(btm_features['q5']) < -1.6).sum()
    total_passed_tests += (np.array(me_features['q5']) < -1.6).sum()
    total_passed_tests += (np.array(mom_features['q5']) < -1.6).sum()
    
    # If half of the possible tests not passed
    if total_passed_tests > threshold:
        LOGGER.info("Holdout is the best validation technique.")
        LOGGER.info("You may stop this experiment and choose this validation technique.")
        LOGGER.info("Finished calculation of best validation technique.")
        return
    

    total_passed_tests = 0

    # Stocks with Percentil95 < 1.5
    total_passed_tests += (np.array(btm_features['q95']) < 1.5).sum()
    total_passed_tests += (np.array(me_features['q95']) < 1.5).sum()
    total_passed_tests += (np.array(mom_features['q95']) < 1.5).sum()

    if total_passed_tests < threshold:
        LOGGER.info(
            f"Only {total_passed_tests} tests passed " + 
            f"out of {number_of_stocks*number_of_characteristics}."
            )
        LOGGER.info("Preq-Grow is not the best validation technique you may choose another.")
        LOGGER.info("You may stop this experiment and find another validation technique.")
        LOGGER.info("Finished calculation of best validation technique.")
        return

    total_passed_tests = 0

    # Stocks with Skewness < 0.3
    total_passed_tests += (np.array(btm_features['skew']) < 0.3).sum()
    total_passed_tests += (np.array(me_features['skew']) < 0.3).sum()
    total_passed_tests += (np.array(mom_features['skew']) < 0.3).sum()

    if total_passed_tests > threshold:
        LOGGER.info("Preq-Grow is the best validation technique.")
        LOGGER.info("We can continue our experiment without a problem.")
        LOGGER.info("Finished calculation of best validation technique.")
        return


    LOGGER.info("Preq-Grow isn't the best validation technique.")
    LOGGER.info("We should stop our experiment and find another validation technique")
    LOGGER.info("Finished calculation of best validation technique.")