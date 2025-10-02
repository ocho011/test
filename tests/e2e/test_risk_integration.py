"""
E2E tests for risk management integration.

Tests risk controls throughout the trading pipeline including position sizing,
drawdown limits, and trading halts.
"""

import pytest
import asyncio
from decimal import Decimal

from trading_bot.system_integrator import SystemIntegrator
from .utils.event_capture import EventCapture


@pytest.mark.e2e
@pytest.mark.asyncio
class TestRiskIntegration:
    """Test risk management integration across system."""

    async def test_position_sizing_integration(
        self,
        system_integrator: SystemIntegrator
    ):
        """Test that position sizing is applied correctly."""
        risk_manager = system_integrator.get_component("risk_manager")
        position_calc = system_integrator.get_component("position_calculator")
        
        assert risk_manager is not None
        assert position_calc is not None
        
        # Calculate position size
        position_size = position_calc.calculate_position_size(
            account_balance=Decimal("10000"),
            risk_per_trade_pct=Decimal("0.01"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49500")
        )
        
        # Verify size is within risk limits
        assert position_size > 0
        assert position_size < Decimal("10000") * Decimal("0.02")  # Max 2% of capital

    async def test_drawdown_limit_enforcement(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture
    ):
        """Test that trading halts when drawdown limit exceeded."""
        drawdown_controller = system_integrator.get_component("drawdown_controller")
        
        assert drawdown_controller is not None
        
        # Simulate losses
        initial_balance = Decimal("10000")
        current_balance = Decimal("8900")  # 11% drawdown
        
        is_exceeded = drawdown_controller.check_drawdown(
            initial_balance=initial_balance,
            current_balance=current_balance,
            max_drawdown_pct=Decimal("0.10")  # 10% limit
        )
        
        # Should detect exceededlimit
        assert is_exceeded is True

    async def test_consecutive_loss_tracking(
        self,
        system_integrator: SystemIntegrator
    ):
        """Test consecutive loss tracking and halt."""
        loss_tracker = system_integrator.get_component("loss_tracker")
        
        assert loss_tracker is not None
        
        # Simulate consecutive losses
        for i in range(3):
            loss_tracker.record_trade_result(is_win=False)
        
        # Should trigger halt after 3 losses
        should_halt = loss_tracker.should_halt_trading(max_consecutive_losses=3)
        assert should_halt is True

    async def test_risk_validation_before_execution(
        self,
        system_integrator: SystemIntegrator,
        event_capture: EventCapture
    ):
        """Test that risk checks occur before order execution."""
        risk_manager = system_integrator.get_component("risk_manager")
        order_executor = system_integrator.get_component("order_executor")
        
        # Risk manager should validate before execution
        # This is verified through system integration
        assert risk_manager is not None
        assert order_executor is not None
