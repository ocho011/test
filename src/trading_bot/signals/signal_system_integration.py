"""
Signal System Integration Example

This module demonstrates how all signal system components work together
to create a comprehensive trading signal generation and validation pipeline.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import asyncio
import logging

from .signal_generator import SignalGenerator, SignalGeneratorConfig
from .confluence_validator import ConfluenceValidator, ConfluenceConfig
from .signal_strength_calculator import SignalStrengthCalculator, StrengthConfig
from .bias_filter import BiasFilter, BiasFilterConfig
from .signal_event_publisher import SignalEventPublisher, PublishingConfig
from .signal_validity_manager import SignalValidityManager, SignalValidityConfig

logger = logging.getLogger(__name__)


class IntegratedSignalSystem:
    """
    Integrated signal generation and validation system

    Combines all signal components into a cohesive pipeline:
    1. Signal generation from ICT patterns
    2. Confluence validation across multiple criteria
    3. Signal strength calculation and scoring
    4. Time-based bias filtering
    5. Event publishing and notification
    6. Signal validity and timeout management
    """

    def __init__(
        self,
        generator_config: Optional[SignalGeneratorConfig] = None,
        confluence_config: Optional[ConfluenceConfig] = None,
        strength_config: Optional[StrengthConfig] = None,
        bias_config: Optional[BiasFilterConfig] = None,
        publisher_config: Optional[PublishingConfig] = None,
        validity_config: Optional[SignalValidityConfig] = None
    ):
        # Initialize components
        self.signal_generator = SignalGenerator(generator_config)
        self.confluence_validator = ConfluenceValidator(confluence_config)
        self.strength_calculator = SignalStrengthCalculator(strength_config)
        self.bias_filter = BiasFilter(bias_config)
        self.event_publisher = SignalEventPublisher(publisher_config)
        self.validity_manager = SignalValidityManager(validity_config)

        # Setup validity manager callbacks
        self._setup_validity_callbacks()

        # Signal tracking
        self._processed_signals: Dict[str, Dict[str, Any]] = {}
        self._signal_counter = 0

        # System state
        self._running = False

    def start(self):
        """Start the integrated signal system"""
        if self._running:
            return

        logger.info("Starting integrated signal system")
        self._running = True

        # Start components that need background processing
        self.validity_manager.start()

    def stop(self):
        """Stop the integrated signal system"""
        if not self._running:
            return

        logger.info("Stopping integrated signal system")
        self._running = False

        # Stop background components
        self.validity_manager.stop()

    async def process_market_data(
        self,
        market_data: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process market data through the complete signal pipeline

        Args:
            market_data: Multi-timeframe market data
            metadata: Additional context and metadata

        Returns:
            List of validated and published signals
        """
        if not self._running:
            raise RuntimeError("Signal system is not running")

        logger.debug("Processing market data through signal pipeline")
        published_signals = []

        try:
            # Step 1: Generate raw signals from ICT patterns
            raw_signal = self.signal_generator.generate_signal(market_data)
            if not raw_signal:
                logger.debug("No signals generated from market data")
                return published_signals

            logger.debug(f"Generated raw signal: {raw_signal['signal_type']}")

            # Step 2: Validate confluence across multiple criteria
            confluence_result = self.confluence_validator.validate_signal_confluence(
                raw_signal, market_data
            )

            if not confluence_result.is_valid:
                logger.debug(f"Signal rejected by confluence validation: {confluence_result.rejection_reasons}")
                return published_signals

            # Add confluence data to signal
            raw_signal['confluence'] = {
                'score': confluence_result.total_score,
                'level': confluence_result.level.value,
                'met_criteria': [c.value for c in confluence_result.met_criteria],
                'validation_details': confluence_result.validation_details
            }

            logger.debug(f"Signal passed confluence validation with score: {confluence_result.total_score}")

            # Step 3: Calculate signal strength and scoring
            strength_result = self.strength_calculator.calculate_signal_strength(
                raw_signal, market_data
            )

            # Add strength data to signal
            raw_signal['strength'] = {
                'total_score': strength_result.total_score,
                'category': strength_result.category.value,
                'level': strength_result.level.value,
                'scores': {cat.value: score for cat, score in strength_result.category_scores.items()},
                'risk_reward_ratio': strength_result.risk_reward_ratio
            }

            logger.debug(f"Signal strength calculated: {strength_result.category.value} ({strength_result.total_score})")

            # Step 4: Apply time-based bias filtering
            bias_result = self.bias_filter.filter_signal(raw_signal, metadata or {})

            if not bias_result.is_valid:
                logger.debug(f"Signal filtered out by bias filter: {bias_result.rejection_reasons}")
                return published_signals

            # Add bias filter data to signal
            raw_signal['bias_filter'] = {
                'session': bias_result.session_info.session_type.value if bias_result.session_info else None,
                'bias_direction': bias_result.bias_direction.value if bias_result.bias_direction else None,
                'filter_reasons': bias_result.rejection_reasons,
                'session_strength': bias_result.session_strength
            }

            logger.debug(f"Signal passed bias filtering for {bias_result.session_info.session_type.value if bias_result.session_info else 'unknown'} session")

            # Step 5: Create final validated signal
            validated_signal = self._create_final_signal(
                raw_signal, confluence_result, strength_result, bias_result, metadata
            )

            # Step 6: Register with validity manager
            signal_id = validated_signal['signal_id']
            validity_info = self.validity_manager.register_signal(
                signal_id=signal_id,
                signal_data=validated_signal,
                custom_timeout_minutes=self._calculate_signal_timeout(validated_signal)
            )

            validated_signal['validity'] = {
                'signal_id': signal_id,
                'expires_at': validity_info.expires_at.isoformat(),
                'timeout_minutes': validity_info.adjusted_timeout_minutes,
                'state': validity_info.state.value
            }

            # Step 7: Publish the signal
            publish_success = await self.event_publisher.publish_signal(
                validated_signal, priority=self._determine_signal_priority(validated_signal)
            )

            if publish_success:
                published_signals.append(validated_signal)
                self._processed_signals[signal_id] = validated_signal
                logger.info(f"Successfully processed and published signal {signal_id}")
            else:
                logger.warning(f"Failed to publish signal {signal_id}")

        except Exception as e:
            logger.error(f"Error processing market data: {e}")

        return published_signals

    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get all currently active signals with full details"""
        active_validity_infos = self.validity_manager.get_active_signals()

        active_signals = []
        for validity_info in active_validity_infos:
            if validity_info.signal_id in self._processed_signals:
                signal = self._processed_signals[validity_info.signal_id].copy()
                signal['validity'] = {
                    'signal_id': validity_info.signal_id,
                    'created_at': validity_info.created_at.isoformat(),
                    'expires_at': validity_info.expires_at.isoformat(),
                    'state': validity_info.state.value,
                    'timeout_minutes': validity_info.adjusted_timeout_minutes
                }
                active_signals.append(signal)

        return active_signals

    def mark_signal_executed(
        self,
        signal_id: str,
        execution_data: Dict[str, Any]
    ) -> bool:
        """
        Mark a signal as executed with execution details

        Args:
            signal_id: ID of executed signal
            execution_data: Execution details (price, volume, etc.)

        Returns:
            True if successfully marked as executed
        """
        success = self.validity_manager.mark_signal_executed(signal_id, execution_data)

        if success and signal_id in self._processed_signals:
            # Update local signal record
            self._processed_signals[signal_id]['execution'] = {
                'executed_at': datetime.now().isoformat(),
                **execution_data
            }
            logger.info(f"Marked signal {signal_id} as executed")

        return success

    def cancel_signal(self, signal_id: str, reason: str = "manual_cancellation") -> bool:
        """Cancel an active signal"""
        success = self.validity_manager.cancel_signal(signal_id, reason)

        if success:
            logger.info(f"Cancelled signal {signal_id}: {reason}")

        return success

    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance statistics"""
        validity_stats = self.validity_manager.get_performance_stats()

        # Add signal generation statistics
        total_processed = len(self._processed_signals)

        # Calculate execution rates and timing
        executed_count = sum(
            1 for signal in self._processed_signals.values()
            if 'execution' in signal
        )

        execution_rate = executed_count / total_processed if total_processed > 0 else 0

        return {
            'total_signals_processed': total_processed,
            'executed_signals': executed_count,
            'execution_rate': execution_rate,
            'validity_management': validity_stats,
            'active_signals_count': len(self.validity_manager.get_active_signals()),
            'expired_signals_count': len(self.validity_manager.get_expired_signals())
        }

    def _create_final_signal(
        self,
        raw_signal: Dict[str, Any],
        confluence_result,
        strength_result,
        bias_result,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create the final validated signal with all component data"""
        self._signal_counter += 1
        signal_id = f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._signal_counter:04d}"

        final_signal = {
            'signal_id': signal_id,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',

            # Core signal data
            'signal_type': raw_signal['signal_type'],
            'direction': raw_signal['direction'],
            'entry_price': raw_signal['entry_price'],
            'stop_loss': raw_signal['stop_loss'],
            'take_profit': raw_signal['take_profit'],
            'timeframe': raw_signal['timeframe'],
            'symbol': raw_signal.get('symbol', 'UNKNOWN'),

            # Pattern information
            'patterns': raw_signal.get('patterns', []),
            'primary_pattern': raw_signal.get('primary_pattern'),

            # Validation results
            'confluence': raw_signal['confluence'],
            'strength': raw_signal['strength'],
            'bias_filter': raw_signal['bias_filter'],

            # Risk management
            'risk_reward_ratio': strength_result.risk_reward_ratio,
            'confidence_score': raw_signal.get('confidence', 0.5),
            'position_size_suggestion': self._calculate_position_size(raw_signal, strength_result),

            # Metadata
            'market_conditions': raw_signal.get('market_conditions', {}),
            'additional_metadata': metadata or {}
        }

        return final_signal

    def _calculate_signal_timeout(self, signal: Dict[str, Any]) -> int:
        """Calculate appropriate timeout for signal based on its characteristics"""
        base_timeout = 60  # 1 hour default

        # Adjust based on timeframe
        timeframe = signal.get('timeframe', '1h')
        timeframe_multipliers = {
            '1m': 0.25, '5m': 0.5, '15m': 0.75, '30m': 1.0,
            '1h': 1.0, '4h': 2.0, '1d': 4.0
        }

        multiplier = timeframe_multipliers.get(timeframe, 1.0)

        # Adjust based on signal strength
        strength_score = signal['strength']['total_score']
        if strength_score > 0.8:
            multiplier *= 1.5  # High strength signals get more time
        elif strength_score < 0.4:
            multiplier *= 0.75  # Low strength signals get less time

        return int(base_timeout * multiplier)

    def _determine_signal_priority(self, signal: Dict[str, Any]):
        """Determine publishing priority based on signal characteristics"""
        from .signal_event_publisher import EventPriority

        strength_score = signal['strength']['total_score']
        confluence_score = signal['confluence']['score']

        # High priority for strong signals with good confluence
        if strength_score > 0.75 and confluence_score > 0.7:
            return EventPriority.HIGH
        elif strength_score > 0.5 and confluence_score > 0.5:
            return EventPriority.NORMAL
        else:
            return EventPriority.LOW

    def _calculate_position_size(
        self,
        signal: Dict[str, Any],
        strength_result
    ) -> Dict[str, Any]:
        """Calculate suggested position size based on signal characteristics"""
        base_risk = 0.02  # 2% base risk

        # Adjust based on strength
        strength_multiplier = strength_result.total_score

        # Adjust based on risk-reward ratio
        rr_multiplier = min(strength_result.risk_reward_ratio / 2.0, 2.0)

        suggested_risk = base_risk * strength_multiplier * rr_multiplier
        suggested_risk = max(0.005, min(0.05, suggested_risk))  # Clamp between 0.5% and 5%

        return {
            'suggested_risk_percentage': suggested_risk,
            'risk_justification': f"Base: {base_risk}, Strength: {strength_multiplier:.2f}, RR: {rr_multiplier:.2f}",
            'max_risk_percentage': 0.05,
            'min_risk_percentage': 0.005
        }

    def _setup_validity_callbacks(self):
        """Setup callbacks for signal validity state changes"""
        from .signal_validity_manager import SignalState

        def on_signal_expired(validity_info):
            logger.info(f"Signal {validity_info.signal_id} expired: {validity_info.validity_reason.value if validity_info.validity_reason else 'unknown'}")

        def on_signal_executed(validity_info):
            logger.info(f"Signal {validity_info.signal_id} executed successfully")

        def on_signal_archived(validity_info):
            # Clean up from local storage when archived
            if validity_info.signal_id in self._processed_signals:
                del self._processed_signals[validity_info.signal_id]
            logger.debug(f"Signal {validity_info.signal_id} archived and cleaned up")

        self.validity_manager.register_state_callback(SignalState.EXPIRED, on_signal_expired)
        self.validity_manager.register_state_callback(SignalState.EXECUTED, on_signal_executed)
        self.validity_manager.register_state_callback(SignalState.ARCHIVED, on_signal_archived)


# Example usage and testing
async def example_usage():
    """Example of how to use the integrated signal system"""

    # Create system with default configurations
    signal_system = IntegratedSignalSystem()

    try:
        # Start the system
        signal_system.start()

        # Example market data (simplified)
        market_data = {
            '1m': pd.DataFrame({
                'timestamp': [datetime.now()],
                'open': [1.2000],
                'high': [1.2010],
                'low': [1.1990],
                'close': [1.2005],
                'volume': [1000]
            }),
            '5m': pd.DataFrame({
                'timestamp': [datetime.now()],
                'open': [1.1995],
                'high': [1.2015],
                'low': [1.1985],
                'close': [1.2005],
                'volume': [5000]
            })
        }

        # Process market data
        signals = await signal_system.process_market_data(
            market_data,
            metadata={'symbol': 'EURUSD', 'session': 'london'}
        )

        print(f"Generated {len(signals)} signals")

        # Get active signals
        active_signals = signal_system.get_active_signals()
        print(f"Active signals: {len(active_signals)}")

        # Get performance stats
        stats = signal_system.get_system_performance()
        print(f"System performance: {stats}")

        # Example of executing a signal
        if signals:
            signal_id = signals[0]['signal_id']
            execution_success = signal_system.mark_signal_executed(
                signal_id,
                {'price': 1.2005, 'volume': 10000, 'execution_type': 'market'}
            )
            print(f"Signal execution marked: {execution_success}")

    finally:
        # Always stop the system
        signal_system.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())