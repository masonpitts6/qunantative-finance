import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from data_api import download_ticker_data


class CapitalAssetPricingModel:

    def __init__(self, tickers, start_date, end_date, market_ticker='SPY', risk_free_rate=0.02/12):
        self.ticker_data = download_ticker_data(tickers, start_date=start_date, end_date=end_date, interval='1mo')['Adj Close']
        self.market_data = download_ticker_data(market_ticker, start_date=start_date, end_date=end_date, interval='1mo')['Adj Close']
        self.tickers = tickers
        self.market_ticker = market_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.RISK_FREE_RATE = risk_free_rate
        self.market_excess_returns = self.calculate_log_returns(self.market_data - self.RISK_FREE_RATE)
        self.stock_excess_returns = self.calculate_log_returns(self.ticker_data - self.RISK_FREE_RATE)


    def calculate_returns(self, price_data):
        """
        Calculate daily returns
        :return: pandas DataFrame
        """
        daily_returns = price_data.pct_change()
        return daily_returns[1:]

    def calculate_log_returns(self, price_data):
        """
        Calculate log returns
        :return: pandas DataFrame
        """
        log_returns = np.log(price_data / price_data.shift(1))
        return log_returns[1:]

    def calculate_market_beta(self):
        """
        Calculate the market beta of a security
        :return: float
        """
        covariance = self.stock_excess_returns.cov(self.market_excess_returns)
        variance = self.market_excess_returns.var()
        return covariance / variance

    def calculate_regression(self):
        """
        Calculate the regression of the stock returns against the market returns
        :return: pandas DataFrame
        """
        beta, alpha = np.polyfit(self.market_excess_returns, self.stock_excess_returns, deg=1)
        print(f"Beta: {beta}")
        print(f"Alpha: {alpha}")
        return beta, alpha

    def plot_regression(self):
        """
        Plot the regression of the stock excess returns against the market excess returns
        :return: None
        """
        beta, alpha = self.calculate_regression()
        fig, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(self.market_excess_returns, self.stock_excess_returns, label=f'{self.tickers} Excess Returns')
        axis.plot(self.market_excess_returns, beta * self.market_excess_returns + alpha, color='red', label='CAPM Line')
        plt.title(f"CAPM Regression of {self.tickers} against {self.market_ticker}")
        plt.xlabel(f"{self.market_ticker} Excess Returns")
        plt.ylabel(f"{self.tickers} Excess Returns")
        # Latex in string to display the generalized CAPM equation
        plt.text(0.08, 0.05, r'$ERP = \alpha + \beta x  MRP$', fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # Testing the CapitalAssetPricingModel class
    ticker_symbols = 'AAPL'
    s_date = '2018-01-01'
    e_date = '2019-12-31'
    capm = CapitalAssetPricingModel(ticker_symbols, s_date, e_date)
    print(capm.calculate_market_beta())
    capm.plot_regression()