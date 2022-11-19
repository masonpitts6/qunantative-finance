import numpy as np
import matplotlib.pyplot as plt


def vasicek_model(r0, kappa, theta, sigma, T=1, N=1_000):

    dt = T / float(N)
    t = np.linspace(0, T, N)
    rates = [r0]

    for _ in range(N - 1):
        dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates.append(rates[-1] + dr)

    return t, rates

def plot_model(t, rates):
    plt.plot(t, rates, label="Vasicek model")
    plt.xlabel("Time(t)")
    plt.ylabel("Interest rate r(t)")
    plt.title("Vasicek model sample path")
    plt.show()


if __name__ == "__main__":
    time, data = vasicek_model(0.05, 0.15, 0.05, 0.01)
    plot_model(time, data)