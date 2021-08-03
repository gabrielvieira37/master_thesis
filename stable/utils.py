"""
"""
import matplotlib.pyplot as plt
import logging
import numpy as np


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