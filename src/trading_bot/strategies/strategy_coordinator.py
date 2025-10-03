"""
Strategy Coordinator for Multi-Timeframe Analysis and Strategy Execution

This module coordinates ICT analysis across multiple timeframes and manages
strategy execution based on confluence conditions.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from trading_bot.analysis import AnalysisResult, ICTAnalyzer, TrendDirection
from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import (
    CandleClosedEvent,
    EventPriority,
    EventType,
    SignalEvent,
    SignalType,
)
from trading_bot.strategies.strategy_system_integration import (
    IntegratedStrategySystem,
)

logger = logging.getLogger(__name__)


class StrategyCoordinator:
    """
    Coordinates multi-timeframe ICT analysis and strategy execution.

    Responsibilities:
    1. Subscribe to CandleClosedEvent for all configured intervals
    2. Run ICT analysis for each timeframe when candles close
    3. Maintain multi-timeframe analysis state
    4. Check confluence conditions across timeframes
    5. Execute strategy and publish SignalEvents when conditions are met
    """

    def __init__(
        self,
        event_bus: EventBus,
        ict_analyzer: ICTAnalyzer,
        strategy_system: IntegratedStrategySystem,
        subscribed_intervals: Optional[List[str]] = None,
        min_confluence_timeframes: int = 2,
    ):
        """
        Initialize StrategyCoordinator.

        Args:
            event_bus: Event bus for subscribing/publishing events
            ict_analyzer: ICT analyzer for technical analysis
            strategy_system: Integrated strategy system for signal generation
            subscribed_intervals: List of intervals to monitor (default: ["5m", "15m", "4h", "1d"])
            min_confluence_timeframes: Minimum timeframes needed for confluence (default: 2)
        """
        self.event_bus = event_bus
        self.ict_analyzer = ict_analyzer
        self.strategy_system = strategy_system
        self.subscribed_intervals = subscribed_intervals or ["5m", "15m", "4h", "1d"]
        self.min_confluence_timeframes = min_confluence_timeframes

        # Store analysis results per timeframe
        self.timeframe_analysis: Dict[str, AnalysisResult] = {}

        # Track subscription IDs for cleanup
        self._subscription_ids: List[str] = []

        # Running state
        self._running = False

        logger.info(
            f"StrategyCoordinator initialized for intervals: {self.subscribed_intervals}"
        )

    async def start(self) -> None:
        """
        Start the coordinator by subscribing to CandleClosedEvent.
        """
        if self._running:
            logger.warning("StrategyCoordinator already running")
            return

        logger.info("Starting StrategyCoordinator...")

        # Subscribe to CandleClosedEvent with high priority
        subscription_id = await self.event_bus.subscribe(
            event_type=EventType.CANDLE_CLOSED,
            handler=self._handle_candle_closed,
            priority=EventPriority.HIGH,
        )
        self._subscription_ids.append(subscription_id)

        self._running = True
        logger.info("StrategyCoordinator started successfully")

    async def stop(self) -> None:
        """
        Stop the coordinator and unsubscribe from events.
        """
        if not self._running:
            logger.warning("StrategyCoordinator not running")
            return

        logger.info("Stopping StrategyCoordinator...")

        # Unsubscribe from all events
        for subscription_id in self._subscription_ids:
            await self.event_bus.unsubscribe(subscription_id)

        self._subscription_ids.clear()
        self._running = False

        logger.info("StrategyCoordinator stopped")

    async def _handle_candle_closed(self, event: CandleClosedEvent) -> None:
        """
        Handle CandleClosedEvent by running ICT analysis and checking confluence.

        Args:
            event: CandleClosedEvent containing symbol, interval, and OHLCV data
        """
        try:
            # Only process intervals we're subscribed to
            if event.interval not in self.subscribed_intervals:
                return

            logger.info(
                f"Processing candle close for {event.symbol} on {event.interval}"
            )

            # Run ICT analysis for this timeframe
            analysis_result = await self._run_ict_analysis(event.df, event.interval)

            # Store analysis result
            self.timeframe_analysis[event.interval] = analysis_result

            logger.debug(
                f"ICT analysis complete for {event.interval}: "
                f"{len(analysis_result.order_blocks)} OBs, "
                f"{len(analysis_result.fair_value_gaps)} FVGs, "
                f"{len(analysis_result.market_structures)} structures"
            )

            # Check confluence across timeframes
            confluence_met = self._check_confluence()

            if confluence_met:
                logger.info(
                    f"Confluence conditions met for {event.symbol}, generating signals"
                )
                await self._generate_and_publish_signals(event.symbol, event.df)
            else:
                logger.debug(
                    f"Confluence not met for {event.symbol} "
                    f"({len(self.timeframe_analysis)}/{self.min_confluence_timeframes} timeframes)"
                )

        except Exception as e:
            logger.error(
                f"Error handling candle closed event for {event.symbol} "
                f"on {event.interval}: {e}",
                exc_info=True,
            )

    async def _run_ict_analysis(
        self, df: pd.DataFrame, timeframe: str
    ) -> AnalysisResult:
        """
        Run ICT analysis for a specific timeframe.

        Args:
            df: OHLCV DataFrame
            timeframe: Timeframe interval

        Returns:
            AnalysisResult containing all ICT patterns and indicators
        """
        # Run analysis in executor to avoid blocking
        loop = asyncio.get_event_loop()
        analysis_result = await loop.run_in_executor(
            None, self.ict_analyzer.analyze_comprehensive, df, timeframe
        )

        return analysis_result

    def _check_confluence(self) -> bool:
        """
        Check if confluence conditions are met across multiple timeframes.

        Confluence requirements:
        1. At least min_confluence_timeframes have analysis results
        2. Multiple timeframes show aligned trend direction
        3. Overlapping support/resistance from order blocks or FVGs

        Returns:
            True if confluence conditions are met, False otherwise
        """
        # Need minimum number of timeframes analyzed
        if len(self.timeframe_analysis) < self.min_confluence_timeframes:
            return False

        # Extract trend directions from market structures
        trends = []
        for interval, result in self.timeframe_analysis.items():
            if result.market_structures:
                # Get most recent market structure trend
                latest_structure = result.market_structures[-1]
                trends.append(latest_structure.direction)

        # Need at least min_confluence_timeframes with trend data
        if len(trends) < self.min_confluence_timeframes:
            return False

        # Check for trend alignment (majority consensus)
        bullish_count = sum(1 for t in trends if t == TrendDirection.BULLISH)
        bearish_count = sum(1 for t in trends if t == TrendDirection.BEARISH)

        # Require majority alignment (>50% of analyzed timeframes)
        aligned = max(bullish_count, bearish_count) > (len(trends) / 2)

        if aligned:
            logger.debug(
                f"Confluence detected: {bullish_count} bullish, "
                f"{bearish_count} bearish across {len(trends)} timeframes"
            )

        return aligned

    async def _generate_and_publish_signals(
        self, symbol: str, df: pd.DataFrame
    ) -> None:
        """
        Generate trading signals using strategy system and publish SignalEvents.

        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame for signal generation
        """
        try:
            # Generate signals using integrated strategy system
            signals = await self.strategy_system.generate_signals(df)

            logger.info(f"Generated {len(signals)} signals for {symbol}")

            # Convert and publish each signal as SignalEvent
            for signal in signals:
                signal_event = self._convert_to_signal_event(symbol, signal)
                await self.event_bus.publish(signal_event)

                logger.debug(
                    f"Published signal: {signal_event.signal_type} "
                    f"with confidence {signal_event.confidence:.2f}"
                )

        except Exception as e:
            logger.error(
                f"Error generating/publishing signals for {symbol}: {e}",
                exc_info=True,
            )

    def _convert_to_signal_event(self, symbol: str, signal: Dict) -> SignalEvent:
        """
        Convert strategy signal to SignalEvent.

        Args:
            symbol: Trading symbol
            signal: Signal dictionary from strategy system

        Returns:
            SignalEvent ready for publishing
        """
        # Map signal direction to SignalType
        signal_type_map = {
            "BUY": SignalType.BUY,
            "SELL": SignalType.SELL,
            "LONG": SignalType.BUY,
            "SHORT": SignalType.SELL,
        }

        signal_direction = signal.get("direction", "").upper()
        signal_type = signal_type_map.get(signal_direction, SignalType.BUY)

        return SignalEvent(
            source="StrategyCoordinator",
            symbol=symbol,
            signal_type=signal_type,
            confidence=signal.get("confidence", 0.5),
            entry_price=signal.get("entry_price"),
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
            quantity=signal.get("quantity"),
            strategy_name=signal.get("strategy_name", "IntegratedStrategy"),
            reasoning=signal.get("reasoning"),
            timestamp=datetime.utcnow(),
            priority=EventPriority.HIGH,
        )

    def get_timeframe_analysis(self, interval: str) -> Optional[AnalysisResult]:
        """
        Get stored analysis result for a specific timeframe.

        Args:
            interval: Timeframe interval

        Returns:
            AnalysisResult if available, None otherwise
        """
        return self.timeframe_analysis.get(interval)

    def get_all_timeframe_analysis(self) -> Dict[str, AnalysisResult]:
        """
        Get all stored analysis results.

        Returns:
            Dictionary mapping intervals to analysis results
        """
        return self.timeframe_analysis.copy()

    def is_running(self) -> bool:
        """
        Check if coordinator is running.

        Returns:
            True if running, False otherwise
        """
        return self._running

    def get_status(self) -> Dict:
        """
        Get coordinator status and statistics.

        Returns:
            Status dictionary with running state and analysis counts
        """
        return {
            "running": self._running,
            "subscribed_intervals": self.subscribed_intervals,
            "min_confluence_timeframes": self.min_confluence_timeframes,
            "analyzed_timeframes": list(self.timeframe_analysis.keys()),
            "timeframe_count": len(self.timeframe_analysis),
            "subscription_count": len(self._subscription_ids),
        }
