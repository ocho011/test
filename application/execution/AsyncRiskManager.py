import asyncio
import logging

from domain.ports.EventBus import EventBus
from domain.events.TradeDecisionEvent import TradeDecisionEvent
from domain.events.OrderEvent import ApprovedTradeOrderEvent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AsyncRiskManager:
    """Manages overall portfolio risk, position sizing, and emergency stop-losses."""
    def __init__(self, event_bus: EventBus, account_balance: float = 10000.0, risk_per_trade: float = 0.01):
        self.event_bus = event_bus
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade  # Risk 1% of account balance per trade
        self.open_positions = {}
        self.total_risk_exposure = 0.0

    async def start_risk_monitoring(self):
        """Starts the risk manager by subscribing to trade decision events."""
        logger.info("Risk Manager started.")
        await self.event_bus.subscribe("PRELIMINARY_TRADE_DECISION", self._handle_trade_decision)
        logger.info("Risk Manager subscribed to PRELIMINARY_TRADE_DECISION.")

    async def _handle_trade_decision(self, event: TradeDecisionEvent):
        """Handles a preliminary trade decision and decides whether to approve it."""
        logger.info(f"Risk Manager received trade decision: {event.decision} {event.symbol}")

        # 1. Calculate Position Size
        risk_amount = self.account_balance * self.risk_per_trade
        stop_loss_delta = abs(event.entry_price - event.stop_loss)
        
        if stop_loss_delta == 0:
            logger.warning("Stop loss cannot be the same as entry price. Trade rejected.")
            return

        position_size = risk_amount / stop_loss_delta

        # 2. Check against total portfolio risk (simplified)
        # In a real system, you'd track total margin used, etc.
        if self.total_risk_exposure + risk_amount > self.account_balance * 0.1: # Max 10% total exposure
            logger.warning(f"Trade rejected for {event.symbol}. Exceeds max portfolio risk.")
            return

        logger.info(f"Trade Approved: {event.decision} {position_size:.4f} {event.symbol} at {event.entry_price}")

        # 3. Publish Approved Trade Order Event
        approved_order = ApprovedTradeOrderEvent(
            symbol=event.symbol,
            order_type="LIMIT",  # Or determine based on strategy
            quantity=position_size,
            decision=event.decision,
            entry_price=event.entry_price,
            stop_loss=event.stop_loss,
            take_profit=event.take_profit
        )
        await self.event_bus.publish(approved_order)
        logger.info(f"Published approved order for {event.symbol}.")

    async def emergency_close_all_positions(self):
        logger.warning("EMERGENCY: Closing all positions!")
        # Logic to quickly liquidate all open positions would go here.
        await asyncio.sleep(0.5)
        logger.warning("All positions closed.")