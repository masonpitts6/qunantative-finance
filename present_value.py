from math import exp


def future_value_discrete(principal, rate, periods):
    return principal * (1 + rate) ** periods


def present_value_discrete(principal, rate, periods):
    return principal * (1 + rate) ** -periods


def future_value_continuous(principal, rate, time):
    return principal * exp(rate * time)


def present_value_continuous(principal, rate, time):
    return principal * exp(-rate * time)


if __name__ == "__main__":
    principal = 100.0

    interest_rate = 0.05

    periods = 10

    print(f"Discrete Future Value of {principal}: {future_value_discrete(principal, interest_rate, periods)}")
    print(f"Discrete Present Value of {principal}: {present_value_discrete(principal, interest_rate, periods)}")
    print(f"Continuous Future Value of {principal}: {future_value_continuous(principal, interest_rate, periods)}")
    print(f"Continuous Present Value of {principal}: {present_value_continuous(principal, interest_rate, periods)}")
