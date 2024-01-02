from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class TradeOrders:
    order_id: int
    date: datetime
    target_weights: pd.Series
    assets: List[str] = None
    quantities:  = None
    order_type: str
    price = price


