import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Number of interest rate processes to simulate
NUM_OF_SIMULATIONS = 1000
# Number of points in a single interest rate process
NUM_OF_POINTS = 1000

def monte_carlo_simulation(x, r0, kappa, theta, sigma, T=1.0):
    """
    Generate a Monte Carlo simulation of interest rate processes
    :param x: float - initial price of the bond
    :param r0: float - initial interest rate
    :param kappa:
    :param theta:
    :param sigma:
    :param T:
    :return:
    """

    dt = T / float(NUM_OF_POINTS)
    result = []

    for _ in range(NUM_OF_SIMULATIONS):
        rates = [r0]
        for _ in range(NUM_OF_POINTS):
            dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)
        result.append(rates)

    simulation_data = pd.DataFrame(result).T

    # Calculate the integral of the interest rate process
    integral_sum = simulation_data.sum() * dt
    # Calculate the bond prices
    present_integral_sum = np.exp(-integral_sum)
    # Mean of bond prices
    bond_price = present_integral_sum.mean()

    print(f"Mean bond price:${bond_price.round(2):,}")


if __name__ == "__main__":
    monte_carlo_simulation(1000, 0.5, 0.3, 0.9, 0.03)