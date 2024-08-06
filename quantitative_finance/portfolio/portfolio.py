import numpy as np
import datetime as dt
from dataclasses import dataclass, field
from typing import List
import pandas as pd
from quantitative_finance.backtest import bt_nb
from quantitative_finance.backtest import statistics as bts
from quantitative_finance.backtest.date_utils import PeriodDateCalculator
from typing import Dict, List
from datetime import datetime


@dataclass
class TradesByPosition:
    position: str  # The name or ticker of the position
    entry_date: np.datetime64  # The date when the position was entered
    exit_date: np.datetime64  # The date when the position was exited
    entry_weight: float  # The portfolio weight at the entry date
    exit_weight: float  # The portfolio weight at the exit date
    entry_price: float  # The price at the entry date
    exit_price: float  # The price at the exit date
    return_: float  # The total return of the trade (could be a percentage or absolute value)
    holding_period: int  # The number of days the position was held
    trade_amount: float = None  # The notional amount traded (optional, if relevant to your analysis)


@dataclass
class TradesByDate:
    start_date: datetime
    end_date: datetime
    weights: np.ndarray
    position_return_contributions: np.ndarray
    portfolio_returns: np.ndarray

    def cumulative_returns(self) -> np.ndarray:
        return np.cumprod(1 + self.portfolio_returns) - 1

    def total_return(self) -> float:
        return self.cumulative_returns()[-1]

    def annualized_return(self, days_per_year: int = 252) -> float:
        total_return = self.total_return()
        num_days = len(self.portfolio_returns)
        return (1 + total_return) ** (days_per_year / num_days) - 1


@dataclass
class Portfolio:
    """

    """
    positions: List[str]
    position_returns: pd.DataFrame
    weights_target: pd.DataFrame
    id: int = None
    name: str = None
    principal: float = 1.0
    dates: np.ndarray = None
    benchmark_returns: pd.DataFrame = None
    return_frequency: str = None
    risk_free_rate: float = 0.0
    weight_drift: bool = True
    normalize_weights: bool = True
    calc_stats: bool = True
    stat_periods: dict = None
    stats: dict = None
    trades: Dict[int, TradesByDate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position_returns_arr = self.position_returns.to_numpy(dtype='float64')
        self.dates = self.position_returns.index.to_numpy(dtype='datetime64[ns]')
        self.weights_target_arr = self.weights_target.to_numpy(dtype='float64')
        self.trade_dates_arr = self.weights_target.index.to_numpy(dtype='datetime64[ns]')
        self.trade_indices_by_position = self.get_trade_indices_by_position()
        self.trade_indices_by_date = self.get_trade_indices()

        self.backtest = Backtest(
            portfolio=self
        )

        if self.calc_stats:
            self.bt_stats = BacktestStatistics(
                backtest=self.backtest
            )

        self.create_trades_by_date()

    def get_trade_indices(
            self,
    ) -> List[int]:
        """
        Finds the indices in the dates array that correspond to the given trade dates.
        If a trade date is not found, it uses the next available date.

        Args:
            dates (np.ndarray): A 1D numpy array of all dates in the portfolio.
            trade_dates (np.ndarray): A 1D numpy array of dates when trades occur.

        Returns:
            List[int]: A list of indices in the dates array that correspond to trade dates
                       or the next available date.

        Example:
            >>> dates = np.array(['2023-07-28', '2023-07-31', '2023-08-01', '2023-08-02'], dtype='datetime64[ns]')
            >>> trade_dates = np.array(['2023-07-29', '2023-07-31', '2023-08-01'], dtype='datetime64[ns]')
            >>> get_trade_indices(dates, trade_dates)
            [1, 1, 2]
        """
        trade_indices = np.searchsorted(self.dates, self.trade_dates_arr, side='left')

        # Ensure indices are within bounds
        trade_indices = np.minimum(trade_indices, len(self.dates) - 1)

        return trade_indices.tolist()

    def get_trade_indices_by_position(
            self,
    ) -> Dict[int, List[int]]:
        """
        Identifies the indices where trades occur for each position in a portfolio.

        This function analyzes a 2D array of target weights over time, identifying
        the rows (time points) where the weights change for each column (position).
        The first row is always included as a trade point.

        Returns:
            Dict[int, List[int]]: A dictionary where keys are column indices
            (positions) and values are lists of row indices (time points)
            where trades occur for that position.

        Example:
            >>> weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.6, 0.4], [0.6, 0.4]])
            >>> self.get_trade_indices_by_position(weights)
            {0: [0, 2], 1: [0, 2]}
        """
        # Get non-zero elements of the diff array, including the first row
        diff = np.vstack([self.weights_target_arr[0], np.diff(self.weights_target_arr, axis=0)])
        non_zero_mask = (diff != 0)

        # Use np.where to get the indices of non-zero elements
        rows, cols = np.where(non_zero_mask)

        return {col: rows[cols == col].tolist() for col in range(self.weights_target_arr.shape[1])}

    def create_trades_by_date(
            self,
    ):
        trade_indices = self.trade_indices_by_date

        if trade_indices[0] != 0:
            trade_indices = [0] + trade_indices
        if trade_indices[-1] != len(self.weights_target_arr):
            trade_indices.append(len(self.weights_target_arr))

        for i, (start, end) in enumerate(zip(trade_indices[:-1], trade_indices[1:])):
            self.trades[i] = TradesByDate(
                start_date=self.dates[start],
                end_date=self.dates[end - 1],
                weights=self.weights_target_arr[start:end],
                position_return_contributions=self.backtest.position_return_contributions_arr[start:end],
                portfolio_returns=self.backtest.portfolio_returns_arr[start:end]
            )

    def get_trade(self, index: int) -> TradesByDate:
        return self.trades[index]

    def get_all_trades(self) -> Dict[int, TradesByDate]:
        return self.trades

    def iterate_trades(self):
        for index, trade in self.trades.items():
            yield index, trade


@dataclass
class Backtest:
    portfolio: Portfolio
    weights_arr: np.array = None
    position_return_contributions_arr: np.array = None
    portfolio_returns_arr: np.array = None
    cumulative_performance_arr: np.array = None

    def __post_init__(self):
        self.run_backtest()

    def run_backtest(
            self,
            portfolio=None
    ):
        portfolio = self.portfolio if portfolio is None else portfolio

        bt_results_tuple = bt_nb.run_backtest(
            position_returns_arr=portfolio.position_returns_arr,
            return_dates=portfolio.dates,
            weights_target_arr=portfolio.weights_target_arr,
            trade_dates=portfolio.trade_dates_arr,
            weight_drift=portfolio.weight_drift
        )

        self.weights_arr = bt_results_tuple[0]
        self.position_return_contributions_arr = bt_results_tuple[1]
        self.portfolio_returns_arr = bt_results_tuple[2]
        self.cumulative_performance_arr = bt_results_tuple[3]


@dataclass
class BacktestStatistics:
    backtest: Backtest

    def __post_init__(self):
        self.periods = PeriodDateCalculator(
            portfolio_dates=self.backtest.portfolio.dates,
            periods=self.backtest.portfolio.stat_periods,
            return_frequency=self.backtest.portfolio.return_frequency,
            fixed_start=False
        )
