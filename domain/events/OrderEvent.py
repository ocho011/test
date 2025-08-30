from dataclasses import dataclass, field
import time
from typing import Optional

@dataclass
class ApprovedTradeOrderEvent:
    """Event representing a trade that has been approved by the risk manager."""
    symbol: str
    order_type: str
    quantity: float
    decision: str
    entry_price: float
    stop_loss: float
    take_profit: float
    event_type: str = "APPROVED_TRADE_ORDER"
    timestamp: float = field(default_factory=time.time)

@dataclass
class OrderStateChangeEvent:
    """Event to signal a change in an order's state."""
    order_id: str
    symbol: str
    new_state: str
    event_type: str = "ORDER_STATE_CHANGE"
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    timestamp: float = field(default_factory=time.time)