import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt


def sim_ornstein_uhlenbeck_process(dt=0.1, theta=0.15, mu=0, sigma=0.1, iterations=10_000):
    """
    Generate a sample path of an Ornstein-Uhlenbeck process
    :param dt: float - time step
    :param theta: float - mean reversion rate
    :param mu: float - mean
    :param sigma: float - standard deviation
    :param iterations: int - number of iterations
    :return: ndarray - sample path
    """
    process = np.zeros(iterations)
    for t in range(1, iterations):
        process[t] = process[t - 1] + theta * (mu - process[t - 1]) * dt + sigma * np.sqrt(dt) * normal()
    return process


def plot_process(process):
    """
    Plot the sample path of an Ornstein-Uhlenbeck process
    :param process: ndarray - sample path
    :return: None
    """
    plt.plot(process, label="Ornstein-Uhlenbeck process")
    plt.xlabel("Time(t)")
    plt.ylabel("Ornstein-Uhlenbeck-process W(t)")
    plt.title("Ornstein-Uhlenbeck-process sample path")
    plt.show()


if __name__ == "__main__":
    process = sim_ornstein_uhlenbeck_process()
    plot_process(process)