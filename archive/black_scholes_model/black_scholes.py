import numpy as np
from scipy import stats


def call_option_price(stock_price, strike_price, time_to_expiration, risk_free_rate, volatility):
    """
    Calculate the price of a call option using the Black-Scholes model
    :param stock_price: float - current stock price
    :param strike_price: float - strike price
    :param time_to_expiration: float - time to maturity in years
    :param risk_free_rate: float - risk-free interest rate
    :param volatility: float - volatility of the underlying stock
    :return: float - price of the call option
    """
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiration) / (
            volatility * np.sqrt(time_to_expiration))
    d2 = d1 - volatility * np.sqrt(time_to_expiration)
    print(f"d1: {d1}")
    print(f"d2: {d2}")
    return stock_price * stats.norm.cdf(d1) - strike_price * np.exp(
        -risk_free_rate * time_to_expiration) * stats.norm.cdf(d2)


def put_option_price(stock_price, strike_price, time_to_expiration, risk_free_rate, volatility):
    """
    Calculate the price of a put option using the Black-Scholes model
    :param stock_price: float - current stock price
    :param strike_price: float - strike price
    :param time_to_expiration: float - time to maturity in years
    :param risk_free_rate: float - risk-free interest rate
    :param volatility: float - volatility of the underlying stock
    :return: float - price of the call option
    """
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiration) / (
            volatility * np.sqrt(time_to_expiration))
    d2 = d1 - volatility * np.sqrt(time_to_expiration)
    print(f"d1: {d1}")
    print(f"d2: {d2}")
    return -stock_price * stats.norm.cdf(-d1) + strike_price * np.exp(
        -risk_free_rate * time_to_expiration) * stats.norm.cdf(-d2)


if __name__ == "__main__":
    # Underlying stock price
    stock_p = 100
    # Strike price
    strike_p = 100
    # Expiry in years
    expiry = 1
    # Risk-free interest rate
    rf = 0.05
    # Volatility
    vol = 0.2

    print(f"Call option price: {call_option_price(stock_p, strike_p, expiry, rf, vol):.2f}")
    print(f"Put option price: {put_option_price(stock_p, strike_p, expiry, rf, vol):.2f}")
