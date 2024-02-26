import numpy as np
from quantitative_finance.data import data_api as data
from scipy.stats import norm
import datetime as dt


# Value at Risk tomorrow with a 95% confidence interval (n=1)
def calc_var(value, mu, sigma, confidence_level=0.95, n=1):
    """
    Calculate the value at risk of a given stock
    :param value: pandas DataFrame - stock data
    :param mu: float - drift of position or mean of returns
    :param sigma: float - standard deviation of returns
    :param confidence_level: float - confidence level
    :param n: int - number of days
    :return: float - value at risk
    """
    return value * (mu * n - sigma * np.sqrt(n) * norm.ppf(1 - confidence_level))

def calc_cvar(value, mu, sigma, confidence_level=0.95, n=1):
    """
    Calculate the conditional value at risk of a given stock
    :param value: pandas DataFrame - stock data
    :param mu: float - drift of position or mean of returns
    :param sigma: float - standard deviation of returns
    :param confidence_level: float - confidence level
    :param n: int - number of days
    :return: float - conditional value at risk
    """
    return -value * (mu * n - sigma * np.sqrt(n) * norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level))


class VaRMonteCarlo:

    def __init__(self, principal, mu, sigma, confidence_level=0.95, n=1, iterations=1000):
        self.principal = principal
        self.mu = mu
        self.sigma = sigma
        self.confidence_level = confidence_level
        self.n = n
        self.iterations = iterations

    def simulate(self, num_of_simulations=1000):
        """
        Simulate the value at risk of a given stock
        :param num_of_simulations: int - number of simulations
        :return: pandas DataFrame - simulated value at risk
        """
        rand = np.random.normal(0, 1, [1, self.iterations])

        # Equation for geometric brownian motion
        # S(t) = S(0) * exp((mu - 0.5 * sigma^2) * t + sigma * W(t))
        price = self.principal * np.exp(self.n * (self.mu - 0.5 * self.sigma ** 2) +
                                        self.sigma * np.sqrt(self.n) * rand)
        # Calculate the percentile of the simulated price to get the maximum loss at the given confidence level
        price = np.sort(price)
        percentile = np.percentile(price, (1 - self.confidence_level) * 100)
        # Calculate the value at risk
        return self.principal - percentile


if __name__ == '__main__':

    start = dt.datetime(2006, 1, 1)
    end = dt.datetime(2022, 1, 1)

    # Get the data
    stock_data = data.download_ticker_data(['SPY'], start, end)['Adj Close']

    # Size of the investment
    p_value = 100_000
    # Calculate the daily returns under the assumption they are normally distributed
    p_returns = stock_data.pct_change().dropna()
    p_mu = np.mean(p_returns)
    p_sigma = np.std(p_returns)

    # Confidence level
    c_level = 0.99

    # Calculate the value at risk
    print(f"Value at risk for ${p_value:,} investment: ${calc_var(p_value, p_mu, p_sigma, c_level).round(2):,}")

    # Calculate the conditional value at risk
    print(f"Conditional value at risk for ${p_value:,} investment: ${calc_cvar(p_value, p_mu, p_sigma, c_level).round(2):,}")

    # Simulate the value at risk
    var_mc = VaRMonteCarlo(principal=p_value, mu=p_mu, sigma=p_sigma, confidence_level=c_level, n=1, iterations=100_000)
    print(f"Simulated value at risk for ${p_value:,} investment: ${var_mc.simulate().round(2):,}")