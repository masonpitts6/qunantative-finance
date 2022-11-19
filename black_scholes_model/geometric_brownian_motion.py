import matplotlib.pyplot as plt
import numpy as np


def sim_geometric_brownian_motion(S0, T=2, n=1000, mu=0, sigma=0.1):
    """
    Generate a sample path of a geometric Brownian motion
    :param S0: float - initial stock price
    :param T:
    :param n: int - number of steps
    :param mu: int or float - drift or mean of the random variable
    :param sigma: int or float - standard deviation of the random variable
    :return:
    """
    dt = T / n
    t = np.linspace(0, T, n)
    W = np.random.standard_normal(size=n)
    # N(0, dt) is equivalent to N(0, 1) * sqrt(dt)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)
    return t, S


def plot_process(t, S):
    """
    Plot the sample path of a geometric Brownian motion
    :param t: ndarray - time steps
    :param S: ndarray - sample path
    :return: None
    """
    plt.plot(t, S, label="Geometric Brownian motion")
    plt.xlabel("Time(t)")
    plt.ylabel("Stock Price S(t)")
    plt.title("Geometric Brownian Motion")
    plt.show()


if __name__ == "__main__":
    time, data = sim_geometric_brownian_motion(10)
    plot_process(time, data)
