import numpy as np
import datetime as dt
from dataclasses import dataclass
from typing import List


@dataclass
class Portfolio:
    """

    """
    positions: List[str]
    position_returns: np.ndarray
    weights_target: np.ndarray
    id: int = None
    name: str = None
    principal: float = 1.0
    dates: List[dt.date] = None
    weight_drift: bool = True
    returns: np.ndarray = None
    benchmark_returns: np.ndarray = None
    return_frequency: str = None
    rebalance_events: List[dt.datetime] = None
    risk_free_rate: float = 0.0
    normalize_weights: bool = True
    calc_stats: bool = True
    stat_periods: dict = None
    stats: dict = None

    def __post_init__(self) -> None:
        pass


