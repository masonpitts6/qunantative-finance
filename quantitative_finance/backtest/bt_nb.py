import numpy as np
import numba


@numba.njit(cache=True)
def run_backtest(
        position_returns_arr,
        return_dates,
        weights_target_arr,
        trade_dates,
        weight_drift=True
):
    if weight_drift:
        weights_arr = calc_portfolio_drift_weights(
            position_returns_arr=position_returns_arr,
            return_dates=return_dates,
            weights_target_arr=weights_target_arr,
            trade_dates=trade_dates
        )
    else:
        weights_arr = weights_target_arr

    position_return_contributions_arr = calc_position_return_contributions(
        position_returns_arr=position_returns_arr,
        weights_arr=weights_arr
    )

    portfolio_returns_arr = calc_portfolio_returns_from_contributions(
        position_contributions_arr=position_return_contributions_arr
    )

    cumulative_performance_arr = calc_cumulative_performance(
        portfolio_returns_arr=portfolio_returns_arr
    )

    return (
        weights_arr,
        position_return_contributions_arr,
        portfolio_returns_arr,
        cumulative_performance_arr
    )


@numba.njit(cache=True)
def calc_portfolio_drift_weights(
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

    return weights_arr


@numba.njit(cache=True)
def calc_position_return_contributions(
        position_returns_arr,
        weights_arr,
):
    weights_arr = weights_arr
    position_contributions = weights_arr * position_returns_arr
    return position_contributions


@numba.njit(cache=True)
def calc_portfolio_returns_from_contributions(
        position_contributions_arr
):
    portfolio_return = np.sum(position_contributions_arr, axis=1)
    return portfolio_return


@numba.njit(cache=True)
def calc_cumulative_performance(
        portfolio_returns_arr
):
    log_returns_arr = np.log(portfolio_returns_arr + 1)
    cumulative_performance = np.exp(np.cumsum(log_returns_arr)) - 1
    return cumulative_performance
