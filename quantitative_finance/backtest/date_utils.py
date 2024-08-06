import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from collections import OrderedDict


class PeriodDateCalculator:
    DAYS_IN_YEAR = 365
    TRADING_DAYS_IN_YEAR = 252
    MONTHS_IN_YEAR = 12
    QUARTERS_IN_YEAR = 4
    YEARS_IN_YEAR = 1

    def __init__(
            self,
            portfolio_dates: np.ndarray,
            return_frequency: str = 'M',
            fixed_start: bool = False,
            periods: Optional[OrderedDict[str, Optional[float]]] = None
    ):
        """
        Initializes the PeriodDateCalculator with the given portfolio dates and options.

        Args:
            portfolio_dates (np.ndarray): Array of portfolio dates.
            return_frequency (str): Frequency of the return calculation, either 'D' for daily or 'M' for monthly.
            fixed_start (bool): If True, periods will be calculated with a fixed start date. If False, periods will be
                calculated with a fixed end date.
            periods (Optional[OrderedDict[str, Optional[float]]]): An ordered dictionary of period names and their
                corresponding lengths in years.
        """
        self.portfolio_dates: pd.DatetimeIndex = pd.to_datetime(portfolio_dates)
        self.return_frequency: str = return_frequency
        self.fixed_start: bool = fixed_start

        if periods is None:
            self.periods = self._default_periods()
        else:
            self.periods = periods

        self.periods_in_year = self._get_periods_in_year()

        if return_frequency not in ['D', 'M', 'Q', 'A']:
            self._return_frequency_error(value=return_frequency)

    def _return_frequency_error(
            self,
            value
    ):
        raise ValueError(
            f'"return_frequency" is {value}, which is not a valid value. Please enter "D", "M", "Q", or "A"')

    def _get_periods_in_year(
            self,
            compounding_periods=False
    ):
        if self.return_frequency == 'D':
            if compounding_periods:
                return self.TRADING_DAYS_IN_YEAR
            return self.DAYS_IN_YEAR
        elif self.return_frequency == 'M':
            return self.MONTHS_IN_YEAR
        elif self.return_frequency == 'Q':
            return self.QUARTERS_IN_YEAR
        elif self.return_frequency == 'A':
            return self.YEARS_IN_YEAR
        else:
            self._return_frequency_error(value=self.return_frequency)

    def calc_period_dates(
            self,
            periods: Optional[OrderedDict[str, Union[float, None]]] = None
    ) -> OrderedDict[str, OrderedDict[str, Union[Tuple[pd.Timestamp, pd.Timestamp], Tuple[int, int]]]]:
        """
        Calculates the start and end dates for various periods based on the portfolio dates.

        Args:
            periods (Optional[OrderedDict[str, Union[float, None]]]): An ordered dictionary of period names and their
                corresponding lengths in years. If None, default periods will be used.

        Returns:
            OrderedDict[str, OrderedDict[str, Union[Tuple[pd.Timestamp, pd.Timestamp], Tuple[int, int]]]]: An ordered
                dictionary where keys are period names, and values are dictionaries containing 'Dates' and 'Indices'
                corresponding to the start and end of the period.
        """
        if periods is None:
            periods = self.periods

        adjusted_periods = self._adjust_periods(periods)
        return self._calculate_period_dates(adjusted_periods)

    @staticmethod
    def _default_periods() -> OrderedDict[str, Optional[float]]:
        """
        Provides the default periods for calculation.

        Returns:
            OrderedDict[str, Optional[float]]: An ordered dictionary of default period names and their lengths in years.
        """
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
            self,
            periods: OrderedDict[str, Optional[float]]
    ) -> OrderedDict[str, Optional[float]]:
        """
        Adjusts the periods based on the return frequency.

        Args:
            periods (OrderedDict[str, Optional[float]]): An ordered dictionary of period names and their lengths in years.

        Returns:
            OrderedDict[str, Optional[float]]]: An ordered dictionary with periods adjusted to the return frequency (daily or monthly).
        """
        factor = self.periods_in_year
        return OrderedDict((k, v * factor if v is not None else None) for k, v in periods.items())

    def _calculate_period_dates(
            self,
            periods: OrderedDict[str, Optional[float]]
    ) -> OrderedDict[str, OrderedDict[str, Union[Tuple[pd.Timestamp, pd.Timestamp], Tuple[int, int]]]]:
        """
        Calculates the period dates based on the adjusted periods.

        Args:
            periods (OrderedDict[str, Optional[float]]): An ordered dictionary of period names and their lengths, adjusted for the return frequency.

        Returns:
            OrderedDict[str, OrderedDict[str, Union[Tuple[pd.Timestamp, pd.Timestamp], Tuple[int, int]]]]: An ordered dictionary of period names, with calculated start and end dates and their corresponding indices.
        """
        period_dates = OrderedDict()
        start_date = self.portfolio_dates[0]
        end_date = self.portfolio_dates[-1]
        compounding_period = self._get_periods_in_year(compounding_periods=True)

        for period_name, adjusted_period_length in periods.items():
            date_offset = self._get_date_offset(adjusted_period_length) if adjusted_period_length is not None else None

            if period_name in ['Inception', 'YTD']:
                if period_name == 'Inception':
                    calculated_start_date = start_date
                else:
                    calculated_start_date = self._find_closest_date_and_index(
                        target_date=pd.Timestamp(f'{end_date.year}-01-01')
                    )[0]
                calculated_end_date = end_date
            else:
                calculated_end_date = start_date + date_offset if self.fixed_start else end_date
                calculated_start_date = end_date - date_offset if not self.fixed_start else start_date

            period_length = (calculated_end_date - calculated_start_date).days / self.DAYS_IN_YEAR

            closest_start_date, start_index = self._find_closest_date_and_index(
                target_date=calculated_start_date
            )
            closest_end_date, end_index = self._find_closest_date_and_index(
                target_date=calculated_end_date
            )

            period_dates[period_name] = OrderedDict([
                ('Dates', (closest_start_date, closest_end_date)),
                ('Indices', (start_index, end_index)),
                ('Number of Years', period_length),
                ('Compounding Periods in Year', compounding_period)
            ])

        return period_dates

    def _find_closest_date_and_index(
            self,
            target_date: pd.Timestamp
    ) -> Tuple[pd.Timestamp, int]:
        """
        Finds the closest date in the portfolio dates to the target date and its index.

        Args:
            target_date (pd.Timestamp): The target date to find in the portfolio dates.

        Returns:
            Tuple[pd.Timestamp, int]: The closest date and its index in the portfolio dates.
        """
        closest_date_idx = np.argmin(np.abs(self.portfolio_dates - target_date))
        return self.portfolio_dates[closest_date_idx], closest_date_idx

    def _get_date_offset(
            self,
            period_length: Optional[float]
    ) -> pd.DateOffset:
        """
        Generates a date offset based on the period length and return frequency.

        Args:
            period_length (Optional[float]): The length of the period in the appropriate time unit (days or months).

        Returns:
            pd.DateOffset: The calculated date offset.

        Raises:
            ValueError: If the period length is None.
        """
        if period_length is None:
            raise ValueError("Period length cannot be None for calculating start date.")

        offset_int = int(period_length)

        if self.return_frequency == 'D':
            date_offset = pd.DateOffset(days=offset_int)
        elif self.return_frequency == 'M':
            date_offset = pd.DateOffset(months=offset_int)
        elif self.return_frequency == 'Q':
            date_offset = pd.DateOffset(months=3 * offset_int)
        elif self.return_frequency == 'A':
            date_offset = pd.DateOffset(years=offset_int)
        else:
            date_offset = None

        return date_offset
