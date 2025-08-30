import asyncio
import logging
from collections import deque
from typing import Dict, Any

from domain.ports.EventBus import EventBus
from domain.events.FVGEvent import FVGEvent
from domain.events.LiquidityEvent import LiquidityEvent
from domain.events.OrderBlockEvent import OrderBlockEvent
from domain.events.TradeDecisionEvent import TradeDecisionEvent
from domain.entities.OrderBlock import OrderBlockType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AsyncStrategyCoordinator:
    """
    Coordinates signals from various analysis components to generate a preliminary trading decision.
    """
    def __init__(self, event_bus: EventBus, max_events: int = 100):
        self.event_bus = event_bus
        # Store recent events for confluence analysis, capped by max_events
        self.recent_fvg: Dict[str, deque] = {}
        self.recent_ob: Dict[str, deque] = {}
        self.recent_liquidity: Dict[str, deque] = {}
        self.max_events = max_events

    async def start_strategy_coordination(self):
        """
        Starts the coordinator by subscribing to relevant analysis events.
        """
        logger.info("Strategy Coordinator started.")
        await self.event_bus.subscribe("NEW_FVG_DETECTED", self._handle_fvg_event)
        await self.event_bus.subscribe("NEW_ORDER_BLOCK", self._handle_ob_event)
        await self.event_bus.subscribe("LIQUIDITY_SWEPT", self._handle_liquidity_event)
        logger.info("Successfully subscribed to analysis events.")

    def _get_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"

    async def _handle_fvg_event(self, event: FVGEvent):
        """Handles new Fair Value Gap events."""
        key = self._get_key(event.symbol, event.timeframe)
        if key not in self.recent_fvg:
            self.recent_fvg[key] = deque(maxlen=self.max_events)
        self.recent_fvg[key].append(event)
        logger.info(f"Handled FVG event for {key}.")
        await self._evaluate_scenarios(event)

    async def _handle_ob_event(self, event: OrderBlockEvent):
        """Handles new Order Block events."""
        key = self._get_key(event.data['symbol'], event.data['timeframe'])
        if key not in self.recent_ob:
            self.recent_ob[key] = deque(maxlen=self.max_events)
        self.recent_ob[key].append(event)
        logger.info(f"Handled Order Block event for {key}.")
        await self._evaluate_scenarios(event)

    async def _handle_liquidity_event(self, event: LiquidityEvent):
        """Handles new Liquidity Sweep events."""
        # Assuming liquidity events are not timeframe specific for now
        key = self._get_key(event.pool.symbol, "N/A")
        if key not in self.recent_liquidity:
            self.recent_liquidity[key] = deque(maxlen=self.max_events)
        self.recent_liquidity[key].append(event)
        logger.info(f"Handled Liquidity event for {key}.")
        await self._evaluate_scenarios(event)

    async def _evaluate_scenarios(self, latest_event: Any):
        """
        Evaluates trading scenarios based on the latest event and recent market context.
        """
        # --- Scenario 1: Bullish OB + Bullish FVG Confluence ---
        # A bullish order block was recently formed, and now a bullish FVG appears,
        # suggesting a potential entry.

        if isinstance(latest_event, FVGEvent):
            key = self._get_key(latest_event.symbol, latest_event.timeframe)
            # Check if there's a recent bullish order block
            if key in self.recent_ob and self.recent_ob[key]:
                last_ob_event = self.recent_ob[key][-1]
                
                # Check for time proximity (e.g., within the last 5 minutes)
                time_diff = latest_event.timestamp - last_ob_event.timestamp
                
                if last_ob_event.order_block.block_type == OrderBlockType.BULLISH and 0 < time_diff < 300:
                    logger.info(f"SCENARIO MET: Bullish OB and FVG confluence for {key}.")
                    
                    # Define trade parameters based on the events
                    entry_price = latest_event.gap.high  # Entry at the top of the FVG
                    stop_loss = last_ob_event.order_block.low  # SL below the OB
                    take_profit = entry_price + (entry_price - stop_loss) * 2  # Simple 1:2 RR
                    
                    decision_event = TradeDecisionEvent(
                        event_type="PRELIMINARY_TRADE_DECISION",
                        symbol=latest_event.symbol,
                        decision="LONG",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        confidence_score=0.75, # Base confidence for this scenario
                        triggering_events=[latest_event, last_ob_event]
                    )
                    await self.event_bus.publish(decision_event)
                    logger.info(f"Published LONG decision for {latest_event.symbol}.")

        # Add more scenarios for SHORT positions and other confluences here...
        pass