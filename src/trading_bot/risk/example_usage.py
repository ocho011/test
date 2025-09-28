"""
Example usage of the risk management system.

This script demonstrates how to use the integrated risk management components
for trading operations.
"""

import asyncio
from datetime import datetime
from decimal import Decimal

from .consecutive_loss_tracker import TradeRecord, TradeResult
from .risk_manager import RiskManager, TradeRequest


async def demo_risk_management():
    """Demonstrate risk management system functionality."""

    # Initialize risk manager
    risk_manager = RiskManager(
        max_risk_score=0.75,  # 75% max risk
        risk_score_weights={
            "drawdown": 0.3,
            "consecutive_losses": 0.25,
            "volatility": 0.25,
            "position_risk": 0.2,
        },
    )

    # Start the system
    await risk_manager.start()

    try:
        print("=== Risk Management System Demo ===\n")

        # Set initial balance
        initial_balance = Decimal("10000.00")
        risk_manager.update_balance(initial_balance)
        print(f"Initial balance: ${initial_balance}")

        # Update volatility data
        symbol = "BTCUSDT"
        atr_value = Decimal("1250.50")
        price = Decimal("45000.00")
        risk_manager.update_volatility(symbol, atr_value, price)
        print(f"Updated volatility for {symbol}: ATR ${atr_value}")

        # Example 1: Normal trade request
        print("\n=== Example 1: Normal Trade Assessment ===")
        trade_request = TradeRequest(
            symbol=symbol,
            entry_price=Decimal("45000.00"),
            stop_loss_price=Decimal("44100.00"),  # 2% stop loss
            account_balance=initial_balance,
            risk_percentage=0.02,  # 2% risk
            win_rate=0.65,
            avg_win_loss_ratio=1.5,
        )

        assessment = risk_manager.assess_trade_risk(trade_request)
        print(f"Decision: {assessment.decision.value}")
        print(f"Confidence: {assessment.confidence:.1%}")
        print(f"Risk Score: {assessment.risk_score:.1%}")
        if assessment.position_size_result:
            print(f"Position Size: {assessment.position_size_result.position_size:.6f}")
            print(
                f"Position Value: ${assessment.position_size_result.position_value:.2f}"
            )
        print(f"Reasons: {assessment.reasons}")

        # Example 2: High risk trade
        print("\n=== Example 2: High Risk Trade Assessment ===")
        high_risk_request = TradeRequest(
            symbol=symbol,
            entry_price=Decimal("45000.00"),
            stop_loss_price=Decimal("40500.00"),  # 10% stop loss
            account_balance=initial_balance,
            risk_percentage=0.05,  # 5% risk
        )

        assessment = risk_manager.assess_trade_risk(high_risk_request)
        print(f"Decision: {assessment.decision.value}")
        print(f"Risk Score: {assessment.risk_score:.1%}")
        print(f"Reasons: {assessment.reasons}")

        # Example 3: Simulate consecutive losses
        print("\n=== Example 3: Consecutive Loss Simulation ===")
        current_balance = initial_balance

        for i in range(4):
            # Record a losing trade
            loss_amount = Decimal("200.00")
            current_balance -= loss_amount

            trade_record = TradeRecord(
                trade_id=f"LOSS_{i+1}",
                symbol=symbol,
                timestamp=datetime.utcnow(),
                entry_price=Decimal("45000.00"),
                exit_price=Decimal("44100.00"),
                quantity=Decimal("0.1"),
                pnl=-loss_amount,
                pnl_percentage=-0.02,
                result=TradeResult.LOSS,
            )

            risk_manager.record_trade_result(trade_record)
            risk_manager.update_balance(current_balance)

            print(f"Loss {i+1}: Balance ${current_balance}, Loss ${loss_amount}")

            # Assess trade after each loss
            assessment = risk_manager.assess_trade_risk(trade_request)
            print(f"  Decision after loss {i+1}: {assessment.decision.value}")
            print(f"  Risk Score: {assessment.risk_score:.1%}")

            if assessment.decision != assessment.decision.ALLOW:
                print(f"  Trading blocked: {', '.join(assessment.reasons)}")
                break

        # Example 4: System status
        print("\n=== Example 4: System Status ===")
        status = risk_manager.get_system_status()
        print(f"Total assessments: {status['risk_manager']['total_assessments']}")
        print(f"Allowed trades: {status['risk_manager']['allowed_trades']}")
        print(f"Blocked trades: {status['risk_manager']['blocked_trades']}")
        print(f"Block rate: {status['risk_manager']['block_rate']:.1%}")

        # Drawdown status
        drawdown = status["drawdown_controller"]
        if drawdown.get("current_balance"):
            print(f"Current balance: ${drawdown['current_balance']:.2f}")
            print(f"Daily drawdown: {drawdown['daily']['drawdown_percentage']:.1%}")
            print(f"Monthly drawdown: {drawdown['monthly']['drawdown_percentage']:.1%}")

        # Loss tracker status
        loss_tracker = status["loss_tracker"]
        print(f"Consecutive losses: {loss_tracker['current_consecutive_losses']}")
        print(f"Trading halted: {loss_tracker['trading_halted']}")

        # Example 5: Recent assessments
        print("\n=== Example 5: Recent Risk Assessments ===")
        recent = risk_manager.get_recent_assessments(limit=3)
        for i, assessment in enumerate(recent, 1):
            print(f"Assessment {i}:")
            print(f"  Decision: {assessment['decision']}")
            print(f"  Risk Score: {assessment['risk_score']:.1%}")
            print(f"  Confidence: {assessment['confidence']:.1%}")

        # Example 6: Force override (emergency)
        print("\n=== Example 6: Emergency Override ===")
        if status["risk_manager"]["blocked_trades"] > 0:
            print("Forcing trading to resume (emergency override)")
            risk_manager.force_allow_trading("Demo emergency override")

            # Test trade after override
            assessment = risk_manager.assess_trade_risk(trade_request)
            print(f"Decision after override: {assessment.decision.value}")

    finally:
        # Clean shutdown
        await risk_manager.stop()
        print("\n=== Risk Management System Demo Complete ===")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_risk_management())
