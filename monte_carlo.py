import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_OF_SIMULATIONS = 1000

def stock_monte_carlo(start_price, mu, sigma, days):
    """
    Generate a Monte Carlo simulation of stock prices
    :param start_price: float - initial stock price
    :param days: int - number of days
    :param mu: float - drift
    :param sigma: float - standard deviation
    :return: ndarray - simulated stock prices
    """
    result = []

    for _ in range(NUM_OF_SIMULATIONS):
        prices = [start_price]
        prices.extend(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()) for _ in range(days))
        result.append(prices)

    simulation_data = pd.DataFrame(result)
    # Transpose so columns contain each simulation
    simulation_data = simulation_data.transpose()
    simulation_data['mean'] = simulation_data.mean(axis=1)
    print(simulation_data)
    plt.plot(simulation_data)
    plt.show()

if __name__ == "__main__":
    stock_monte_carlo(10, 0.1, 0.2, 10)