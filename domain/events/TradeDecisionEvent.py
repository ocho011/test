from dataclasses import dataclass, field
import time
from typing import List, Any, Dict

@dataclass
class TradeDecisionEvent:
    """Event to represent a preliminary trading decision."""
    event_type: str  # e.g., PRELIMINARY_TRADE_DECISION
    symbol: str
    decision: str  # e.g., "LONG", "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence_score: float
    triggering_events: List[Any] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
