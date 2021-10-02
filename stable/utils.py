"""
"""
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
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


def data_split(size, train_percentage, val_percentage):
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
    val_end_split = int(size*(train_percentage+val_percentage))
    train_index =np.arange(train_size)
    val_index = np.arange(start=train_size, stop=val_end_split)
    test_index = np.arange(start=val_end_split, stop=size)
    indexes_list = [(train_index, val_index, test_index)]
    LOGGER.info(f'Train size: {train_size}, val size: {val_end_split-train_size}, test size: {size-val_end_split}')
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
def constrain_weights(weights):
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
    """

    volumes = pd.read_csv(os.path.join(data_path , 'monthly_volume.csv'))
    comission = 0.0004
    uk_variable1 = 0.002
    uk_variable2 = 0.01

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
    x = x * (uk_variable1 - uk_variable2)

    liq_cost = np.ones_like(x)*uk_variable2 + x
    market_cost += liq_cost
    market_cost = 1-market_cost
    market_cost = market_cost
    return market_cost


class ParametricPortifolioNN(nn.Module):
    def __init__(self, benchmark, return_, risk_constant, number_of_stocks):
        super(ParametricPortifolioNN, self).__init__()
        # self.first_layer = nn.Linear(3,3)
        self.weights = nn.Linear(3,1, bias=False)
        self.relu = nn.ReLU()
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
        x = F.relu(self.weights(x))
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