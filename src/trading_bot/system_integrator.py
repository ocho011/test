"""
System integrator for orchestrating all trading bot components.

This module provides the SystemIntegrator class that manages the initialization,
lifecycle, and coordination of all system components with proper dependency
ordering and configuration management.
"""

import asyncio
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from .core.base_component import BaseComponent
from .core.di_container import DIContainer, Lifetime
from .core.event_bus import EventBus
from .core.lifecycle_manager import ComponentLifecycleManager, StartupOrder
from .core.message_hub import MessageHub
from .config.config_manager import ConfigManager
from .config.models import TradingBotConfig
from .config import setup_logging

# Data layer
from .data import (
    BinanceClient,
    MarketDataProvider,
    MarketDataAggregator,
    DataCache,
    RateLimiter,
)

# Analysis layer
from .analysis import ICTAnalyzer, TechnicalIndicators

# Signal layer
from .signals import (
    SignalGenerator,
    ConfluenceValidator,
    SignalStrengthCalculator,
    BiasFilter,
    SignalEventPublisher,
    SignalValidityManager,
)

# Strategy layer
from .strategies import (
    StrategyRegistry,
    StrategySelector,
    StrategyCoordinator,
    IntegratedStrategySystem,
    ICTStrategy,
    TraditionalIndicatorStrategy,
    StrategyPerformanceTracker,
)

# Execution layer
from .execution import (
    OrderExecutor,
    MarketOrderManager,
    LimitOrderManager,
    PositionTracker,
    PartialTakeProfitManager,
    TrailingStopManager,
    SlippageController,
)

# Risk management
from .risk import (
    RiskManager,
    PositionSizeCalculator,
    DrawdownController,
    ConsecutiveLossTracker,
    VolatilityFilter,
)

# Notifications
from .notifications import NotificationManager

# Monitoring
from .monitoring import HealthMonitor


class SystemIntegrator(BaseComponent):
    """
    Central orchestrator for the entire trading system.
    
    Manages component initialization, dependency injection, lifecycle
    coordination, and system-wide operations for both local and VM
    deployment environments.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        """
        Initialize the system integrator.
        
        Args:
            config_path: Path to configuration file (None for auto-detect)
            environment: Environment name (development, production, testing)
        """
        super().__init__("SystemIntegrator")
        
        # Configuration
        self.config_manager = ConfigManager()
        self.config: Optional[TradingBotConfig] = None
        self.config_path = config_path
        self.environment = environment or "development"
        
        # Core infrastructure
        self.di_container: Optional[DIContainer] = None
        self.event_bus: Optional[EventBus] = None
        self.message_hub: Optional[MessageHub] = None
        self.lifecycle_manager: Optional[ComponentLifecycleManager] = None
        
        # Component references (for direct access if needed)
        self.components: Dict[str, BaseComponent] = {}
        
        # Health monitoring
        self.health_monitor: Optional[HealthMonitor] = None
        
        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._signal_handlers_installed = False

    async def _start(self) -> None:
        """Start the entire system."""
        self.logger.info(f"Starting trading system in {self.environment} mode...")
        
        try:
            # Phase 1: Load configuration
            await self._load_configuration()
            
            # Phase 2: Initialize infrastructure
            await self._initialize_infrastructure()
            
            # Phase 3: Register components
            await self._register_components()
            
            # Phase 4: Start all components
            success = await self.lifecycle_manager.start_all_components()
            
            if not success:
                raise RuntimeError("Failed to start all components")
            
            # Phase 5: Install signal handlers
            self._install_signal_handlers()
            
            self.logger.info("Trading system started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start trading system: {e}")
            await self._emergency_shutdown()
            raise

    async def _stop(self) -> None:
        """Stop the entire system gracefully."""
        self.logger.info("Stopping trading system...")
        
        try:
            # Stop all components in reverse order
            if self.lifecycle_manager:
                await self.lifecycle_manager.stop_all_components()
            
            # Stop infrastructure
            if self.event_bus:
                await self.event_bus.stop()
            
            # Dispose DI container
            if self.di_container:
                self.di_container.dispose()
            
            self.logger.info("Trading system stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    async def _load_configuration(self) -> None:
        """Load and validate configuration."""
        self.logger.info(f"Loading configuration for {self.environment} environment")
        
        self.config = self.config_manager.load_config(
            config_path=self.config_path,
            environment=self.environment
        )
        
        # Initialize proper logging system with LogManager
        setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Configuration loaded successfully from {self.environment}")
        self.logger.info(f"Logs will be written to: {self.config.logging.output_dir}/")

    async def _initialize_infrastructure(self) -> None:
        """Initialize core infrastructure components."""
        self.logger.info("Initializing core infrastructure...")
        
        # Create DI container
        self.di_container = DIContainer(name="trading_bot")
        
        # Create event bus
        self.event_bus = EventBus(max_queue_size=self.config.system.event_queue_size)
        await self.event_bus.start()
        
        # Create message hub
        self.message_hub = MessageHub(
            max_queue_size=100
        )
        await self.message_hub.start()
        
        # Create lifecycle manager
        self.lifecycle_manager = ComponentLifecycleManager(
            di_container=self.di_container,
            message_hub=self.message_hub
        )
        await self.lifecycle_manager.start()
        
        # Create health monitor
        self.health_monitor = HealthMonitor(
            check_interval_seconds=30.0,
            enable_auto_checks=True
        )
        await self.health_monitor.start()
        
        # Register infrastructure in DI container
        self.di_container.register_instance(DIContainer, self.di_container)
        self.di_container.register_instance(EventBus, self.event_bus)
        self.di_container.register_instance(MessageHub, self.message_hub)
        self.di_container.register_instance(ConfigManager, self.config_manager)
        self.di_container.register_instance(TradingBotConfig, self.config)
        self.di_container.register_instance(HealthMonitor, self.health_monitor)
        
        self.logger.info("Core infrastructure initialized")

    async def _register_components(self) -> None:
        """Register all system components with lifecycle manager."""
        self.logger.info("Registering system components...")
        
        # DATA LAYER (StartupOrder.DATA)
        await self._register_data_components()
        
        # ANALYSIS LAYER (StartupOrder.ANALYSIS)
        await self._register_analysis_components()
        
        # SIGNAL LAYER (StartupOrder.ANALYSIS)
        await self._register_signal_components()
        
        # STRATEGY LAYER (StartupOrder.ANALYSIS)
        await self._register_strategy_components()
        
        # EXECUTION LAYER (StartupOrder.EXECUTION)
        await self._register_execution_components()
        
        # RISK LAYER (StartupOrder.EXECUTION)
        await self._register_risk_components()
        
        # NOTIFICATION LAYER (StartupOrder.NOTIFICATION)
        await self._register_notification_components()
        
        # MONITORING LAYER - Register all components with health monitor
        await self._register_monitoring_components()
        
        self.logger.info(f"Registered {len(self.components)} components")

    async def _register_data_components(self) -> None:
        """Register data layer components."""
        # Rate limiter
        from .data.rate_limiter import RateLimit
        default_rate_limit = RateLimit(requests_per_minute=1200, max_burst=50)
        rate_limiter = RateLimiter(default_rate_limit=default_rate_limit)
        self.di_container.register_instance(RateLimiter, rate_limiter)
        
        # Binance client
        binance_client = BinanceClient(
            api_key=self.config.binance.api_key,
            api_secret=self.config.binance.api_secret,
            testnet=self.config.binance.testnet,
            event_bus=self.event_bus
        )
        self.di_container.register_instance(BinanceClient, binance_client)
        self.components["binance_client"] = binance_client
        self.lifecycle_manager.register_component(
            binance_client,
            startup_order=StartupOrder.DATA,
            dependencies=[]
        )
        
        # Data cache (not a BaseComponent, so only register with DI container)
        data_cache = DataCache(
            cache_dir="data/cache",
            max_cache_size_gb=5.0,
            max_age_days=30
        )
        self.di_container.register_instance(DataCache, data_cache)
        self.components["data_cache"] = data_cache
        
        # Market data provider
        # Get symbol and intervals from config
        symbol = self.config.trading.symbol
        timeframe = self.config.trading.timeframe
        
        # Set up default intervals based on timeframe
        # Always include the main timeframe plus one higher timeframe for context
        default_intervals = [timeframe]
        if timeframe not in ["4h", "1d"]:
            # Add a higher timeframe for multi-timeframe analysis
            higher_tf = {"5m": "15m", "15m": "4h", "1h": "4h"}.get(timeframe, "1d")
            if higher_tf not in default_intervals:
                default_intervals.append(higher_tf)
        
        market_data_provider = MarketDataProvider(
            binance_client=binance_client,
            event_bus=self.event_bus,
            default_symbol=symbol,
            default_intervals=default_intervals
        )
        self.di_container.register_instance(MarketDataProvider, market_data_provider)
        self.components["market_data_provider"] = market_data_provider
        self.lifecycle_manager.register_component(
            market_data_provider,
            startup_order=StartupOrder.DATA,
            dependencies=["BinanceClient"]  # data_cache is not a BaseComponent
        )
        
        # Market data aggregator
        # Uses multiple symbols and intervals for multi-symbol tracking
        symbols = [symbol]  # Start with main symbol, can be expanded
        intervals = default_intervals
        
        market_data_aggregator = MarketDataAggregator(
            binance_client=binance_client,
            event_bus=self.event_bus,
            symbols=symbols,
            intervals=intervals,
            lookback_bars=100
        )
        self.di_container.register_instance(MarketDataAggregator, market_data_aggregator)
        self.components["market_data_aggregator"] = market_data_aggregator
        self.lifecycle_manager.register_component(
            market_data_aggregator,
            startup_order=StartupOrder.DATA,
            dependencies=["BinanceClient", "MarketDataProvider"]
        )

    async def _register_analysis_components(self) -> None:
        """Register analysis layer components."""
        # Technical indicators
        technical_indicators = TechnicalIndicators()
        self.di_container.register_instance(TechnicalIndicators, technical_indicators)
        self.components["technical_indicators"] = technical_indicators
        
        # ICT analyzer (not a BaseComponent, so only register with DI container)
        ict_analyzer = ICTAnalyzer(
            enable_validation=True,
            enable_parallel_processing=True,
            memory_optimization=True
        )
        self.di_container.register_instance(ICTAnalyzer, ict_analyzer)
        self.components["ict_analyzer"] = ict_analyzer

    async def _register_signal_components(self) -> None:
        """Register signal generation components."""
        # Get configuration
        signals_config = self.config.signals
        
        # Get ICTAnalyzer for accessing sub-components
        ict_analyzer = self.di_container.resolve(ICTAnalyzer)
        
        # Signal generator - use config-based initialization
        signal_generator = SignalGenerator(
            min_confidence_threshold=signals_config.min_confidence,
            max_signals_per_timeframe=3,  # Could add to config
            pattern_weights=None,  # Use defaults
            risk_reward_ratios=None  # Use defaults
        )
        self.di_container.register_instance(SignalGenerator, signal_generator)
        self.components["signal_generator"] = signal_generator
        
        # Confluence validator - resolve required dependencies from TechnicalAnalyzer
        confluence_validator = ConfluenceValidator(
            order_block_detector=ict_analyzer.order_block_detector,
            fvg_analyzer=ict_analyzer.fvg_analyzer,
            structure_analyzer=ict_analyzer.market_structure_analyzer,
            timeframe_manager=ict_analyzer.timeframe_manager,
            pattern_validator=ict_analyzer.validation_engine,
            config=None  # Use default ConfluenceConfig
        )
        self.di_container.register_instance(ConfluenceValidator, confluence_validator)
        self.components["confluence_validator"] = confluence_validator
        
        # Signal strength calculator - requires pattern validator
        signal_strength_calc = SignalStrengthCalculator(
            pattern_validator=ict_analyzer.validation_engine,
            config=None  # Use default StrengthConfig
        )
        self.di_container.register_instance(SignalStrengthCalculator, signal_strength_calc)
        self.components["signal_strength_calculator"] = signal_strength_calc
        
        # Bias filter - config-based initialization
        bias_filter = BiasFilter(config=None)  # Use default BiasFilterConfig
        self.di_container.register_instance(BiasFilter, bias_filter)
        self.components["bias_filter"] = bias_filter
        
        # Signal event publisher - use config-based initialization
        signal_publisher = SignalEventPublisher(
            config=None,  # Will use default PublishingConfig
            event_handlers=[]  # Add event_bus.publish as handler if needed
        )
        # Add event_bus integration as event handler
        signal_publisher.add_event_handler(lambda event: self.event_bus.publish(event))
        self.di_container.register_instance(SignalEventPublisher, signal_publisher)
        self.components["signal_publisher"] = signal_publisher
        
        # Signal validity manager - config-based initialization
        validity_manager = SignalValidityManager(config=None)  # Use default SignalValidityConfig
        self.di_container.register_instance(SignalValidityManager, validity_manager)
        self.components["signal_validity_manager"] = validity_manager

    async def _register_strategy_components(self) -> None:
        """Register strategy layer components."""
        # Strategy registry
        strategy_registry = StrategyRegistry()
        self.di_container.register_instance(StrategyRegistry, strategy_registry)
        
        # Register strategy classes (not instances)
        strategy_registry.register(ICTStrategy, name="ICT_Strategy", version="1.0.0")
        strategy_registry.register(TraditionalIndicatorStrategy, name="Traditional_Strategy", version="1.0.0")

        # Create strategy instances for immediate use
        ict_strategy = ICTStrategy(name="ICT_Strategy", version="1.0.0", parameters=None)
        self.components["ict_strategy"] = ict_strategy

        traditional_strategy = TraditionalIndicatorStrategy(name="Traditional_Strategy", version="1.0.0", parameters=None)
        self.components["traditional_strategy"] = traditional_strategy
        
        # Strategy selector
        strategy_selector = StrategySelector(registry=strategy_registry)
        self.di_container.register_instance(StrategySelector, strategy_selector)
        self.components["strategy_selector"] = strategy_selector
        
        # Performance tracker
        performance_tracker = StrategyPerformanceTracker()
        self.di_container.register_instance(StrategyPerformanceTracker, performance_tracker)
        self.components["performance_tracker"] = performance_tracker
        
        # Integrated strategy system (no dependencies)
        integrated_strategy_system = IntegratedStrategySystem()
        self.di_container.register_instance(IntegratedStrategySystem, integrated_strategy_system)
        self.components["integrated_strategy_system"] = integrated_strategy_system
        
        # Strategy coordinator - requires ICTAnalyzer, IntegratedStrategySystem, EventBus
        ict_analyzer = self.di_container.resolve(ICTAnalyzer)
        strategy_coordinator = StrategyCoordinator(
            event_bus=self.event_bus,
            ict_analyzer=ict_analyzer,
            strategy_system=integrated_strategy_system,
            subscribed_intervals=None,  # Uses default ["5m", "15m", "4h", "1d"]
            min_confluence_timeframes=2
        )
        self.di_container.register_instance(StrategyCoordinator, strategy_coordinator)
        self.components["strategy_coordinator"] = strategy_coordinator

    async def _register_execution_components(self) -> None:
        """Register execution layer components."""
        # Position tracker
        binance_client = self.di_container.resolve(BinanceClient)
        position_tracker = PositionTracker(
            binance_client=binance_client,
            event_bus=self.event_bus
        )
        self.di_container.register_instance(PositionTracker, position_tracker)
        self.components["position_tracker"] = position_tracker
        self.lifecycle_manager.register_component(
            position_tracker,
            startup_order=StartupOrder.EXECUTION,
            dependencies=["MarketDataProvider"]
        )
        
        # Slippage controller
        from trading_bot.execution.slippage_controller import SlippageConfig
        slippage_config = SlippageConfig()  # Uses default values
        slippage_controller = SlippageController(config=slippage_config)
        self.di_container.register_instance(SlippageController, slippage_controller)
        self.components["slippage_controller"] = slippage_controller
        
        # Order managers
        binance_client = self.di_container.resolve(BinanceClient)
        market_order_manager = MarketOrderManager(
            binance_client=binance_client,
            event_bus=self.event_bus
        )
        self.di_container.register_instance(MarketOrderManager, market_order_manager)

        limit_order_manager = LimitOrderManager(
            binance_client=binance_client,
            event_bus=self.event_bus
        )
        self.di_container.register_instance(LimitOrderManager, limit_order_manager)
        
        # Order executor
        # Get dry_run setting from execution config (default to False if not set)
        dry_run = getattr(self.config.execution, 'dry_run', False) if hasattr(self.config, 'execution') else False
        order_executor = OrderExecutor(
            binance_client=binance_client,
            event_bus=self.event_bus,
            dry_run=dry_run
        )
        self.di_container.register_instance(OrderExecutor, order_executor)
        self.components["order_executor"] = order_executor
        self.lifecycle_manager.register_component(
            order_executor,
            startup_order=StartupOrder.EXECUTION,
            dependencies=["BinanceClient", "PositionTracker"]
        )

        # Trailing stop manager (registered after OrderExecutor)
        trailing_stop_manager = TrailingStopManager(
            order_executor=order_executor,
            event_bus=self.event_bus,
            config=None  # Uses default TrailingStopConfig
        )
        self.di_container.register_instance(TrailingStopManager, trailing_stop_manager)
        self.components["trailing_stop_manager"] = trailing_stop_manager
        self.lifecycle_manager.register_component(
            trailing_stop_manager,
            startup_order=StartupOrder.EXECUTION,
            dependencies=["MarketDataProvider", "PositionTracker", "OrderExecutor"]
        )

        # Partial take profit manager (registered after OrderExecutor)
        partial_tp_manager = PartialTakeProfitManager(
            order_executor=order_executor,
            event_bus=self.event_bus,
            config=None  # Uses default PartialTakeProfitConfig
        )
        self.di_container.register_instance(PartialTakeProfitManager, partial_tp_manager)
        self.components["partial_tp_manager"] = partial_tp_manager

    async def _register_risk_components(self) -> None:
        """Register risk management components."""
        # Position size calculator
        position_calc = PositionSizeCalculator()  # Uses default parameter values
        self.di_container.register_instance(PositionSizeCalculator, position_calc)
        self.components["position_calculator"] = position_calc
        
        # Drawdown controller
        drawdown_controller = DrawdownController()  # Uses default parameter values
        self.di_container.register_instance(DrawdownController, drawdown_controller)
        self.components["drawdown_controller"] = drawdown_controller
        
        # Consecutive loss tracker
        loss_tracker = ConsecutiveLossTracker()
        self.di_container.register_instance(ConsecutiveLossTracker, loss_tracker)
        self.components["loss_tracker"] = loss_tracker
        
        # Volatility filter
        volatility_filter = VolatilityFilter()  # Uses default parameter values
        self.di_container.register_instance(VolatilityFilter, volatility_filter)
        self.components["volatility_filter"] = volatility_filter
        
        # Risk manager (creates its own internal risk components)
        risk_manager = RiskManager(name="RiskManager")  # Use PascalCase name for consistency
        self.di_container.register_instance(RiskManager, risk_manager)
        self.components["risk_manager"] = risk_manager
        self.lifecycle_manager.register_component(
            risk_manager,
            startup_order=StartupOrder.EXECUTION,
            dependencies=["MarketDataProvider"]
        )

    async def _register_notification_components(self) -> None:
        """Register notification components."""
        # Discord bot manager
        from trading_bot.notifications.discord_bot import DiscordBotManager
        discord_manager = DiscordBotManager(config=self.config)
        self.di_container.register_instance(DiscordBotManager, discord_manager)
        self.components["discord_manager"] = discord_manager

        # Notification manager (not a BaseComponent, no lifecycle registration)
        notification_manager = NotificationManager(
            discord_manager=discord_manager,
            config=None  # Uses default NotificationConfig
        )
        self.di_container.register_instance(NotificationManager, notification_manager)
        self.components["notification_manager"] = notification_manager

    async def _register_monitoring_components(self) -> None:
        """Register monitoring components and component health checkers."""
        # Register all components with health monitor
        for name, component in self.components.items():
            if isinstance(component, BaseComponent):
                self.health_monitor.register_component(component)
        
        self.logger.info(f"Registered {len(self.components)} components with health monitor")

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        if self._signal_handlers_installed:
            return
        
        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self._signal_handlers_installed = True
        self.logger.info("Signal handlers installed")

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    async def _emergency_shutdown(self) -> None:
        """Emergency shutdown in case of critical failure."""
        self.logger.critical("Initiating emergency shutdown...")
        
        try:
            if self.lifecycle_manager:
                await self.lifecycle_manager.stop_all_components(force=True)
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")

    async def restart(self) -> None:
        """Restart the entire system."""
        self.logger.info("Restarting trading system...")
        
        await self.stop()
        await asyncio.sleep(2.0)  # Brief pause
        await self.start()

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dict containing system status information
        """
        if not self.lifecycle_manager:
            return {"status": "not_initialized"}
        
        return {
            "status": "running" if self.is_running() else "stopped",
            "environment": self.environment,
            "components": self.lifecycle_manager.get_component_status(),
            "statistics": self.lifecycle_manager.get_system_stats(),
            "event_bus": {
                "running": self.event_bus.is_running() if self.event_bus else False,
                "stats": self.event_bus.get_stats() if self.event_bus else {},
                "queue_sizes": self.event_bus.get_queue_sizes() if self.event_bus else {},
            },
        }

    def get_component(self, name: str) -> Optional[BaseComponent]:
        """
        Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(name)

    async def get_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health status for the entire system.
        
        This is the main health check endpoint that aggregates health
        information from all registered components.
        
        Returns:
            Dict containing system health information in JSON format
        """
        if not self.health_monitor:
            return {
                "status": "unknown",
                "message": "Health monitor not initialized"
            }
        
        system_health = await self.health_monitor.get_health()
        return system_health.to_dict()

    async def get_readiness(self) -> Dict[str, Any]:
        """
        Get readiness status indicating if system can accept traffic.
        
        Returns:
            Dict with readiness status and basic component information
        """
        if not self.health_monitor:
            return {
                "ready": False,
                "message": "Health monitor not initialized"
            }
        
        system_health = await self.health_monitor.get_health()
        
        return {
            "ready": system_health.is_ready,
            "status": system_health.status.value,
            "timestamp": system_health.timestamp.isoformat(),
            "components_summary": {
                "total": len(system_health.components),
                "healthy": sum(
                    1 for c in system_health.components.values()
                    if c.status.value == "healthy"
                ),
                "degraded": sum(
                    1 for c in system_health.components.values()
                    if c.status.value == "degraded"
                ),
                "unhealthy": sum(
                    1 for c in system_health.components.values()
                    if c.status.value == "unhealthy"
                ),
            }
        }

    async def get_liveness(self) -> Dict[str, Any]:
        """
        Get liveness status indicating if system is running.
        
        Simple check to verify the application is alive and responding.
        
        Returns:
            Dict with liveness status
        """
        return {
            "alive": self.is_running(),
            "status": "running" if self.is_running() else "stopped",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            Dict containing system resource and performance metrics
        """
        if not self.health_monitor:
            return {
                "error": "Health monitor not initialized"
            }
        
        metrics = await self.health_monitor.get_metrics()
        return metrics.to_dict()

    async def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """
        Get health status for a specific component.
        
        Args:
            component_name: Name of the component to check
            
        Returns:
            Dict with component health information or error
        """
        if not self.health_monitor:
            return {
                "error": "Health monitor not initialized"
            }
        
        component_health = self.health_monitor.get_component_status(component_name)
        
        if component_health is None:
            return {
                "error": f"Component '{component_name}' not found"
            }
        
        return component_health.to_dict()
