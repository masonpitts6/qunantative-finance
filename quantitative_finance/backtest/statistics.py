import numba
import numpy as np


@numba.njit(cache=True)
def calc_cumulative_performance(
        portfolio_returns_arr
):
    log_returns_arr = np.log(portfolio_returns_arr + 1)
    cumulative_performance = np.exp(np.cumsum(log_returns_arr)) - 1
    return cumulative_performance
