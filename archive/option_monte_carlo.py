import numpy as np


class OptionPricing:

    def __init__(self, stock_price_initial, strike_price, time_to_maturity, volatility, risk_free_rate,
                 iterations=10_000):
        self.stock_price_initial = stock_price_initial
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.iterations = iterations

    def sim_call_option(self):
        option_data = np.zeros([self.iterations, 2])

        rand = np.random.normal(0, 1, [1, self.iterations])

        # Calculate the stock price at expiration
        stock_price_expiration = self.stock_price_initial * np.exp(
            self.time_to_maturity * (self.risk_free_rate - 0.5 * self.volatility ** 2) + self.volatility * np.sqrt(
                self.time_to_maturity) * rand)

        # Calculate the payoff at expiration i.e. the difference between the strike price and the stock price
        option_data[:, 1] = stock_price_expiration - self.strike_price

        # Calculate the average of the monte carlo simulation to get the most probable outcome
        # amax function returns the maximum of each row since the payoff cannot be negative either 0 or positive
        average_payoff = np.sum(np.amax(option_data, axis=1))/float(self.iterations)
        # Calculate the present value of the option payoff
        # Continuously compounded interest rate
        return np.exp(-self.risk_free_rate * self.time_to_maturity) * average_payoff

    def sim_put_option(self):
        option_data = np.zeros([self.iterations, 2])

        rand = np.random.normal(0, 1, [1, self.iterations])

        # Calculate the stock price at expiration
        stock_price_expiration = self.stock_price_initial * np.exp(
            self.time_to_maturity * (self.risk_free_rate - 0.5 * self.volatility ** 2) + self.volatility * np.sqrt(
                self.time_to_maturity) * rand)

        # Calculate the payoff at expiration i.e. the difference between the strike price and the stock price
        option_data[:, 1] = self.strike_price - stock_price_expiration

        # Calculate the average of the monte carlo simulation to get the most probable outcome
        # amax function returns the maximum of each row since the payoff cannot be negative either 0 or positive
        average_payoff = np.sum(np.amax(option_data, axis=1))/float(self.iterations)
        # Calculate the present value of the option payoff
        # Continuously compounded interest rate
        return np.exp(-self.risk_free_rate * self.time_to_maturity) * average_payoff


# TODO: Run simulations to validate the accuracy of the option pricing model
if __name__ == '__main__':
    option = OptionPricing(stock_price_initial=100,
                           strike_price=100,
                           time_to_maturity=1,
                           volatility=0.2,
                           risk_free_rate=0.05,
                           iterations=1_000_000)
    print(f'Expected value of call option: ${option.sim_call_option().round(2)}')
    print(f'Expected value of put option: ${option.sim_put_option().round(2)}')
