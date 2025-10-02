"""
Main entry point for the trading bot system.

This module provides the command-line interface and main execution logic
for running the trading bot in various modes (trading, backtest, paper).
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from .system_integrator import SystemIntegrator


# Configure basic logging (will be replaced by LogManager during system init)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


class TradingBotCLI:
    """Command-line interface for the trading bot."""

    def __init__(self):
        """Initialize the CLI."""
        self.system: Optional[SystemIntegrator] = None
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description='ICT Trading Bot - Automated cryptocurrency trading system',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run in development mode (default)
  python -m trading_bot.main

  # Run in production mode
  python -m trading_bot.main --env production

  # Run in paper trading mode
  python -m trading_bot.main --mode paper

  # Run backtest
  python -m trading_bot.main --mode backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31

  # Run with custom config
  python -m trading_bot.main --config /path/to/config.yml
            """
        )

        # Environment and configuration
        parser.add_argument(
            '--env', '--environment',
            type=str,
            default='development',
            choices=['development', 'production', 'testing'],
            help='Environment to run in (default: development)'
        )

        parser.add_argument(
            '--config',
            type=str,
            help='Path to custom configuration file'
        )

        # Run mode
        parser.add_argument(
            '--mode',
            type=str,
            default='trading',
            choices=['trading', 'paper', 'backtest'],
            help='Execution mode (default: trading)'
        )

        # Backtest options
        parser.add_argument(
            '--symbol',
            type=str,
            default='BTCUSDT',
            help='Trading symbol for backtest (default: BTCUSDT)'
        )

        parser.add_argument(
            '--start',
            type=str,
            help='Backtest start date (YYYY-MM-DD)'
        )

        parser.add_argument(
            '--end',
            type=str,
            help='Backtest end date (YYYY-MM-DD)'
        )

        # System options
        parser.add_argument(
            '--daemon',
            action='store_true',
            help='Run as daemon (background process)'
        )

        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug logging'
        )

        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Dry run mode (no actual trading)'
        )

        # Health check
        parser.add_argument(
            '--health-check',
            action='store_true',
            help='Check system health and exit'
        )

        # Version
        parser.add_argument(
            '--version',
            action='version',
            version='ICT Trading Bot v1.0.0'
        )

        return parser

    async def run_trading_mode(self, args: argparse.Namespace) -> int:
        """
        Run the bot in live or paper trading mode.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        logger.info(f"Starting trading bot in {args.mode} mode...")
        
        try:
            # Create and start system
            self.system = SystemIntegrator(
                config_path=args.config,
                environment=args.env
            )
            
            await self.system.start()
            
            # Wait for shutdown signal
            logger.info("Trading bot is running. Press Ctrl+C to stop.")
            await self.system.wait_for_shutdown()
            
            # Graceful shutdown
            logger.info("Shutting down trading bot...")
            await self.system.stop()
            
            logger.info("Trading bot stopped successfully")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            if self.system:
                await self.system.stop()
            return 0
            
        except Exception as e:
            logger.error(f"Fatal error in trading mode: {e}", exc_info=True)
            if self.system:
                await self.system.stop()
            return 1

    async def run_backtest_mode(self, args: argparse.Namespace) -> int:
        """
        Run the bot in backtest mode.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Exit code
        """
        logger.info("Starting backtest mode...")
        
        try:
            from .backtesting import BacktestEngine
            
            # Validate backtest parameters
            if not args.start or not args.end:
                logger.error("Backtest mode requires --start and --end dates")
                return 1
            
            # Create backtest engine
            engine = BacktestEngine(
                symbol=args.symbol,
                start_date=args.start,
                end_date=args.end,
                initial_capital=10000.0  # Default, will be overridden by config
            )
            
            # Run backtest
            logger.info(f"Running backtest for {args.symbol} from {args.start} to {args.end}")
            results = await engine.run()
            
            # Display results
            logger.info("Backtest completed successfully")
            logger.info(f"Total trades: {results.get('total_trades', 0)}")
            logger.info(f"Win rate: {results.get('win_rate', 0):.2%}")
            logger.info(f"Total return: {results.get('total_return', 0):.2%}")
            logger.info(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max drawdown: {results.get('max_drawdown', 0):.2%}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Fatal error in backtest mode: {e}", exc_info=True)
            return 1

    async def run_health_check(self, args: argparse.Namespace) -> int:
        """
        Run health check and exit.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Exit code
        """
        logger.info("Running health check...")
        
        try:
            # Create system (but don't start fully)
            self.system = SystemIntegrator(
                config_path=args.config,
                environment=args.env
            )
            
            # Load configuration
            await self.system._load_configuration()
            
            # Check configuration
            logger.info("✓ Configuration loaded successfully")
            
            # Check infrastructure
            await self.system._initialize_infrastructure()
            logger.info("✓ Infrastructure initialized successfully")
            
            # Cleanup
            await self.system._stop()
            
            logger.info("✓ Health check passed")
            return 0
            
        except Exception as e:
            logger.error(f"✗ Health check failed: {e}", exc_info=True)
            return 1

    async def run(self, args: argparse.Namespace) -> int:
        """
        Run the application based on arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Exit code
        """
        # Set debug logging if requested
        if args.debug:
            logging.getLogger('trading_bot').setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        # Health check mode
        if args.health_check:
            return await self.run_health_check(args)

        # Backtest mode
        if args.mode == 'backtest':
            return await self.run_backtest_mode(args)

        # Trading or paper mode
        return await self.run_trading_mode(args)

    def main(self) -> int:
        """
        Main entry point.
        
        Returns:
            Exit code
        """
        args = self.parser.parse_args()
        
        # Run async main
        try:
            return asyncio.run(self.run(args))
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            return 1


def main() -> int:
    """
    Main entry point for the trading bot.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    cli = TradingBotCLI()
    return cli.main()


if __name__ == '__main__':
    sys.exit(main())
