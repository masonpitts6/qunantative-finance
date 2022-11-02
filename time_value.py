from math import exp


def future_value_discrete(principal, rate, periods):
    return principal * (1 + rate) ** periods


def present_value_discrete(principal, rate, periods):
    return principal * (1 + rate) ** -periods


def future_value_continuous(principal, rate, time):
    return principal * exp(rate * time)


def present_value_continuous(principal, rate, time):
    return principal * exp(-rate * time)


def future_value_annuity(payment, rate, periods, payment_end=True):
    """
    Calculate the present value of an annuity
    :param payment: int or float
    :param rate: float
    :param periods: int or float
    :return: float
    """
    if payment_end:
        return payment * ((1 + rate) ** periods - 1) / rate
    else:
        return (1 + rate) * payment * ((1 + rate) ** periods - 1) / rate


def present_value_annuity(payment, rate, periods, payment_end=True):
    """
    Calculate the present value of an annuity
    :param payment: int or float
    :param rate: float
    :param periods: int or float
    :return: float
    """
    if payment_end:
        return payment * ((1 - (1 + rate) ** -periods) / rate)
    else:
        return (1 + rate) * payment * ((1 - (1 + rate) ** -periods) / rate)


def future_value_cf(cash_flows, rate, cf_end=True):
    """
    Calculate the future value of a series of cash flows
    :param cash_flows: list of ints or floats
    :param rate: float
    :param cf_end: boolean
    :return: float
    """
    # Cash flows occur at the end of each period
    # i.e. t=1 when the first cash flow occurs, t=2 when the second cash flow occurs etc.
    if cf_end:
        return sum(cash_flows * (1 + rate) ** p for p, cash_flows in enumerate(cash_flows))

    # Cash flows occur at the beginning of each period i.e. t=0 when the first cash flow occurs, t=1 when the second
    # cash flow and no cash flow occurs at t=n (end of period)
    else:
        return sum(cash_flows * (1 + rate) ** p for p, cash_flows in enumerate(cash_flows, 1))


def present_value_cf(cash_flows, rate, cf_end=True):
    """
    Calculate the present value of a series of cash flows
    :param cash_flows: list of ints or floats
    :param rate: float
    :param cf_end: boolean
    :return: float
    """
    # Cash flows occur at the end of each period
    # i.e. t=1 when the first cash flow occurs, t=2 when the second cash flow occurs etc.
    if cf_end:
        return sum(cash_flows / (1 + rate) ** p for p, cash_flows in enumerate(cash_flows, 1))

    # Cash flows occur at the beginning of each period i.e. t=0 when the first cash flow occurs, t=1 when the second
    # cash flow and no cash flow occurs at t=n (end of period)
    else:
        return sum(cash_flows / (1 + rate) ** p for p, cash_flows in enumerate(cash_flows))


if __name__ == "__main__":
    principal = 100.0

    interest_rate = 0.05

    periods = 10

    print(f"Discrete Future Value of {principal}: {future_value_discrete(principal, interest_rate, periods)}")
    print(f"Discrete Present Value of {principal}: {present_value_discrete(principal, interest_rate, periods)}")
    print(f"Continuous Future Value of {principal}: {future_value_continuous(principal, interest_rate, periods)}")
    print(f"Continuous Present Value of {principal}: {present_value_continuous(principal, interest_rate, periods)}")
