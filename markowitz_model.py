import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NFLX', 'NVDA', 'JPM', 'ADBE']

start_date = '2015-01-01'
end_date = '2022-10-31'

NUM_TRADING_DAYS = 252


def download_stock_data(stocks, start_date, end_date):
    """
    Download data from Yahoo Finance
    :param stocks: list of strings
    :param start_date: string
    :param end_date: string
    :return: pandas DataFrame
    """
    return yf.download(stocks, start=start_date, end=end_date)['Adj Close']


def calculate_daily_returns(price_data):
    """
    Calculate daily returns
    :param price_data: pandas DataFrame
    :return: pandas DataFrame
    """
    daily_returns = price_data.pct_change()
    return daily_returns[1:]


def calculate_log_returns(price_data):
    """
    Calculate log returns
    :param price_data: pandas DataFrame
    :return: pandas DataFrame
    """
    return np.log(price_data / price_data.shift(1))


def calculate_cumulative_returns(price_data):
    """
    Calculate cumulative returns
    :param price_data: pandas DataFrame
    :return: pandas DataFrame
    """
    log_return = (price_data / price_data.iloc[0]) - 1
    # Drops the first row of data which is NaN
    return log_return[1:]


def mean_arithmetic_annualized(daily_returns):
    """
    Calculate the arithmetic mean
    :return: float
    """
    return daily_returns.mean() * NUM_TRADING_DAYS


def mean_geometric_annualized(price_data):
    """
    Calculate the geometric mean
    :return: float
    """
    return (calculate_cumulative_returns(price_data).iloc[-1, :] ** (1 / len(price_data))) ** NUM_TRADING_DAYS - 1


def volatility_annualized(daily_returns):
    """
    Calculate volatility
    :param daily_returns: pandas DataFrame
    :return: pandas DataFrame
    """
    return daily_returns.std(axis=0) * np.sqrt(NUM_TRADING_DAYS)


def get_statistics(price_data, log_returns=False):
    """
    Calculate statistics
    :param price_data: pandas DataFrame
    :param log_returns: boolean
    :return: pandas DataFrame
    """
    if log_returns:
        daily_returns = calculate_log_returns(price_data)
    else:
        daily_returns = calculate_daily_returns(price_data)
    stats = pd.DataFrame()
    stats['mean_geometric'] = mean_geometric_annualized(price_data)
    stats['mean_arithmetic'] = mean_arithmetic_annualized(daily_returns)
    stats['median'] = daily_returns.median(axis=0) * NUM_TRADING_DAYS
    stats['std'] = volatility_annualized(daily_returns)
    stats['skew'] = daily_returns.skew(axis=0)
    stats['kurtosis'] = daily_returns.kurtosis(axis=0)
    stats['sharpe'] = daily_returns.mean(axis=0) / daily_returns.std()
    return stats


def calculate_correlation_matrix(price_data):
    """
    Calculate correlation matrix
    :param price_data: pandas DataFrame
    :return: pandas DataFrame
    """
    return price_data.corr()


def plot_correlation_matrix(correlation_matrix):
    """
    Plot correlation matrix
    :param correlation_matrix: pandas DataFrame
    :return: None
    """
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.show()


def plot_line(data):
    """
    Plot stock data
    :param data: pandas DataFrame
    :return: None
    """
    data.plot(figsize=(10, 6))
    plt.show()

def portfolio_mean_variance():
    pass



if __name__ == '__main__':
    data = download_stock_data(stocks, start_date, end_date)
    plot_line(data)
    cumulative_returns = calculate_cumulative_returns(data)
    plot_line(cumulative_returns)
    log_returns = calculate_log_returns(data)
    plot_line(log_returns)
    returns = calculate_daily_returns(data)
    stats = get_statistics(data)
    print(stats)
    corr_matrix = calculate_correlation_matrix(data)
