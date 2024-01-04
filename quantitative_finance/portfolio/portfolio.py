import numpy as np
import datetime as dt
from dataclasses import dataclass
from typing import List
import pandas as pd
from quantitative_finance.backtest import bt_nb


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

    def __post_init__(self) -> None:
        self.position_returns_arr = self.position_returns.to_numpy()
        self.dates = self.position_returns.index.to_numpy()
        self.weights_target_arr = self.weights_target.to_numpy()
        self.trade_dates_arr = self.weights_target.index.to_numpy()

        self.backtest = Backtest(
            portfolio=self
        )


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
