import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr


def wiener_process(mu=0, sigma=0.1, x0=0, n=1000):
    """
    Generate a sample path of a Wiener process (Brownian motion)
    :param sigma: float - variance
    :param x0: float - initial value
    :param n: int - number of steps
    :param mu: float - drift
    :param sigma: float - normal distribution variance
    :return: ndarray - sample path
    """
    W = np.zeros(n+1)

    # Create n+1 time steps
    t = np.linspace(x0, n, n + 1)

    # Cumulative sum of the Wiener process
    # Every step is a random draw from a normal distribution N(0, dt)
    # N(0, dt) is equivalent to N(0, 1) * sqrt(dt)

    W[1:n + 1] = np.cumsum(npr.normal(loc=mu, scale=sigma, size=n))

    return t, W


def plot_process(t, W):
    """
    Plot the sample path of a Wiener process
    :param t: ndarray - time steps
    :param W: ndarray - sample path
    :return: None
    """
    plt.plot(t, W, label="Wiener process")
    plt.xlabel("Time(t)")
    plt.ylabel("Wiener-process W(t)")
    plt.title("Wiener-process sample path")
    plt.show()


if __name__ == "__main__":
    time, data = wiener_process(mu=1, sigma=5)
    plot_process(time, data)
