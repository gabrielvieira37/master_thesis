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


def main():
    data_path = "../data/"
    mcap, lreturn, book_to_mkt_ratio, monthly_return = load_data(data_path)
    stock_names = list(monthly_return.columns)
    firm_characteristics, r, time, number_of_stocks = create_characteristics(mcap, lreturn, book_to_mkt_ratio, monthly_return,stock_names)
    print("Done")


if __name__ == "__main__":
    main()
