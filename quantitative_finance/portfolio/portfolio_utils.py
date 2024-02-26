import numpy as np
import pandas as pd
from typing import Optional, Union
from collections import OrderedDict

DAYS_OF_YEAR = 365.25
MONTHS_OF_YEAR = 12
QUARTERS_OF_YEAR = 4


def calc_period_dates(
        portfolio_dates: np.ndarray,
        periods: Optional[OrderedDict[str, Union[float, None]]] = None,
        return_frequency: str = 'M',
        fixed_start: bool = False
) -> OrderedDict[str, OrderedDict[str, Union[tuple, tuple]]]:
    if return_frequency not in ['D', 'M']:
        raise ValueError("return_frequency must be either 'D' or 'M'.")

    if periods is None:
        periods = _default_periods()

    adjusted_periods = _adjust_periods(
        periods,
        return_frequency
    )
    return _calculate_period_dates(
        portfolio_dates,
        adjusted_periods,
        return_frequency,
        fixed_start
    )


def _default_periods() -> OrderedDict[str, Optional[float]]:
    return OrderedDict([
        ("1M", 1 / 12),
        ("3M", 3 / 12),
        ("6M", 6 / 12),
        ("YTD", None),
        ("1Y", 1),
        ("2Y", 2),
        ("3Y", 3),
        ("5Y", 5),
        ("7Y", 7),
        ("10Y", 10),
        ("15Y", 15),
        ("20Y", 20),
        ("Inception", None)
    ])


def _adjust_periods(
        periods: OrderedDict[str, Optional[float]],
        return_frequency: str
) -> OrderedDict[str, Optional[float]]:
    factor = DAYS_OF_YEAR if return_frequency == 'D' else MONTHS_OF_YEAR
    return OrderedDict((k, v * factor if v is not None else None) for k, v in periods.items())


def _calculate_period_dates(
        portfolio_dates: np.ndarray,
        periods: OrderedDict[str, Optional[float]],
        return_frequency: str,
        fixed_start: bool
) -> OrderedDict[str, OrderedDict[str, Union[tuple, tuple]]]:
    period_dates = OrderedDict()
    start_date = portfolio_dates[0]
    start_date = pd.Timestamp(start_date)
    end_date = portfolio_dates[-1]
    end_date = pd.Timestamp(end_date)

    for period_name, period_length in periods.items():
        date_offset = _get_date_offset(period_length, return_frequency) if period_length is not None else None

        if period_name in ['Inception', 'YTD']:
            if period_name == 'Inception':
                calculated_start_date = start_date
            else:
                calculated_start_date = _find_closest_date_and_index(
                    dates=portfolio_dates,
                    target_date=pd.Timestamp(f'{end_date.year}-01-01')
                )[0]
            calculated_end_date = end_date
        else:
            calculated_end_date = start_date + date_offset if fixed_start else end_date
            calculated_start_date = end_date - date_offset if not fixed_start else start_date

        closest_start_date, start_index = _find_closest_date_and_index(
            dates=portfolio_dates,
            target_date=calculated_start_date
        )
        closest_end_date, end_index = _find_closest_date_and_index(
            dates=portfolio_dates,
            target_date=calculated_end_date
        )

        period_dates[period_name] = OrderedDict([
            ('Dates', (closest_start_date, closest_end_date)),
            ('Indices', (start_index, end_index))
        ])

    return period_dates


def _find_closest_date_and_index(dates: np.ndarray, target_date: pd.Timestamp) -> tuple:
    dates = pd.to_datetime(dates)
    closest_date_idx = np.argmin(np.abs(dates - target_date))
    return dates[closest_date_idx], closest_date_idx


def _get_date_offset(
        period_length: Optional[float],
        return_frequency: str
) -> pd.Timestamp:
    """
    Calculates the start date from the end date and period length.
    ...
    """
    if period_length is None:
        raise ValueError("Period length cannot be None for calculating start date.")

    offset_int = int(period_length)

    if return_frequency == 'D':
        date_offset = pd.DateOffset(days=offset_int)
    elif return_frequency == 'M':
        date_offset = pd.DateOffset(months=offset_int)
    else:
        date_offset = pd.Timestamp()

    return date_offset


