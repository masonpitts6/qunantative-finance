import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import seaborn as sns
import yfinance as yf

stocks = ['SPY', 'AGG', 'EEM', 'VGK']

start_date = '2010-01-01'
end_date = '2022-10-31'

NUM_TRADING_DAYS = 252
NUM_PORTFOLIO = 100_000


def download_stock_data(tickers, start_date, end_date):
    """
    Download data from Yahoo Finance
    :param tickers: list of strings
    :param start_date: string
    :param end_date: string
    :return: pandas DataFrame
    """
    return yf.download(tickers, start=start_date, end=end_date)['Adj Close']


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


def portfolio_mean(returns, weights):
    return np.sum(returns.mean() * weights) * NUM_TRADING_DAYS


def portfolio_volatility(returns, weights):
    return np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)) ** 0.5


def generate_portfolios(returns):
    portfolio_means = []
    portfolio_vols = []
    portfolio_weights = []

    for _ in range(10000):
        weight = np.random.random(len(returns.columns))
        # Normalize weights
        weight /= np.sum(weight)
        portfolio_weights.append(weight)
        portfolio_means.append(portfolio_mean(returns, weight))
        portfolio_vols.append(portfolio_volatility(returns, weight))

    return np.array(portfolio_means), np.array(portfolio_vols), np.array(portfolio_weights)


def plot_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


# Calculate negative sharpe ratio that is used to optimize the portfolio for mean variance.
# Scipy can only minimize functions, not maximize so we need to minimize the negative sharpe ratio to maximize
# the sharpe ratio.
def negative_sharpe(weights, returns):
    """
    Calculate negative sharpe ratio
    :param returns: pandas DataFrame
    :param weights: numpy array
    """
    return -portfolio_mean(returns, weights) / portfolio_volatility(returns, weights)


def optimize_portfolio_sharpe(returns, weights):
    """
    Optimize portfolio for sharpe ratio
    :param returns: pandas DataFrame
    :param weights: pandas DataFrame
    :return:
    """
    # Sum of the weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Weights must be between 0 and 1 since we are using long only positions
    bounds = tuple((0, 1) for _ in range(len(returns.columns)))
    return optimize.minimize(fun=negative_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds,
                             constraints=constraints)


def print_optimal_portfolio(optimum, returns):
    print(f"Optimal portfolio: {optimum['x'].round(3)}")


def plot_optimal_portfolio(optimum, returns, portfolio_returns, portfolio_volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns / portfolio_volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(portfolio_volatility(returns, optimum['x']), portfolio_mean(returns, optimum['x']), 'r*', markersize=15.0)
    plt.show()


if __name__ == '__main__':
    df = download_stock_data(stocks, start_date, end_date)
    plot_line(df)
    cumulative_returns = calculate_cumulative_returns(df)
    plot_line(cumulative_returns)
    log_returns = calculate_log_returns(df)
    plot_line(log_returns)
    s_returns = calculate_daily_returns(df)
    stats = get_statistics(df)
    print(stats)
    corr_matrix = calculate_correlation_matrix(df)
    plot_correlation_matrix(corr_matrix)
    p_means, p_vols, p_weights = generate_portfolios(log_returns)
    plot_portfolios(p_means, p_vols)
    p_optimum = optimize_portfolio_sharpe(log_returns, p_weights)
    plot_optimal_portfolio(p_optimum, log_returns, p_means, p_vols)
