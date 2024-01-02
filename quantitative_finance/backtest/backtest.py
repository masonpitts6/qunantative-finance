import numpy as np
import numba


@numba.njit(cache=True)
def backtest_drift_numba(
        position_returns_arr,
        return_dates,
        weights_target_arr,
        trade_dates
):
    weights_arr = np.zeros(position_returns_arr.shape)

    trade_date_index = 0
    trade_date_length = len(trade_dates)
    trade_date = trade_dates[trade_date_index]

    for i, date in enumerate(return_dates):

        # Check if there is a portfolio trade and execute it
        if date == trade_date or i == 0:

            new_weight = weights_target_arr[trade_date_index]

            if trade_date_index < trade_date_length - 1:
                trade_date_index += 1
                trade_date = trade_dates[trade_date_index]
        else:
            # Grow or shrink current weights by previous period's returns. Generates portfolio weight drift.
            prev_weights = weights_arr[i - 1]
            prev_returns = position_returns_arr[i - 1]
            total_return = np.dot(prev_weights, (1 + prev_returns))
            new_weight = prev_weights * (1 + prev_returns) / total_return

        weights_arr[i] = new_weight

    position_contributions = weights_arr * (position_returns_arr + 1)
    portfolio_returns = np.sum(position_contributions, axis=1)
    cumulative_returns = np.cumprod(portfolio_returns) - 1

    return weights_arr, position_contributions, portfolio_returns, cumulative_returns
