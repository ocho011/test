# Trading Bot - Development Guide

Complete guide for developing, testing, and running the ICT-based cryptocurrency trading bot.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Development Workflows](#development-workflows)
- [Configuration Guide](#configuration-guide)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Quick Start

### Prerequisites

- **Python 3.10+** - Required for modern type hints and features
- **pip** - Python package manager
- **virtualenv** - Virtual environment management (recommended)
- **Git** - Version control
- **Binance Account** - For live/paper trading API access

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd trading-bot
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Configure environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your API credentials
   ```

5. **Run in development mode**:
   ```bash
   ./scripts/run_local.sh
   ```

### First Run Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -e .`)
- [ ] `.env` file created with API credentials
- [ ] API credentials validated (testnet recommended first)
- [ ] Configuration reviewed (`config/environments/development.yml`)

---

## Project Structure

```
trading-bot/
├── src/trading_bot/           # Main application source
│   ├── core/                  # Core infrastructure
│   │   ├── di_container.py    # Dependency injection
│   │   ├── event_bus.py       # Event-driven messaging
│   │   └── lifecycle_manager.py # Component lifecycle
│   ├── data/                  # Market data layer
│   │   ├── binance_client.py  # Binance API integration
│   │   └── market_data_provider.py # Data streaming
│   ├── signals/               # Signal generation
│   │   ├── signal_generator.py # ICT pattern detection
│   │   └── confluence.py      # Signal validation
│   ├── strategies/            # Trading strategies
│   │   ├── strategy_selector.py # Strategy orchestration
│   │   └── ict_strategy.py    # ICT implementation
│   ├── execution/             # Order execution
│   │   ├── order_executor.py  # Order management
│   │   └── position_manager.py # Position tracking
│   ├── risk/                  # Risk management
│   │   ├── risk_manager.py    # Risk validation
│   │   └── portfolio_risk.py  # Portfolio-level risk
│   ├── notifications/         # Alert system
│   │   └── discord_notifier.py # Discord integration
│   ├── config/                # Configuration system
│   │   ├── config_manager.py  # Config loader
│   │   ├── models.py          # Pydantic models
│   │   └── environments/      # Environment configs
│   ├── system_integrator.py   # System orchestrator
│   └── main.py                # CLI entry point
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── data/                  # Test fixtures
├── config/                    # Configuration files
│   ├── environments/          # Environment-specific configs
│   └── trading_config.yaml    # Configuration template
├── scripts/                   # Utility scripts
│   ├── run_local.sh          # Local development
│   ├── run_backtest.sh       # Backtesting
│   └── run_paper.sh          # Paper trading
├── docs/                      # Documentation
│   └── development.md         # This file
├── .env.template              # Environment template
└── pyproject.toml            # Project metadata
```

### Key Modules

#### Core Infrastructure (`core/`)
- **DIContainer**: Dependency injection for loose coupling
- **EventBus**: Event-driven communication between components
- **LifecycleManager**: Manages component startup/shutdown ordering

#### Data Layer (`data/`)
- **BinanceClient**: Low-level Binance API wrapper
- **MarketDataProvider**: High-level market data streaming
- Handles WebSocket connections, data caching, rate limiting

#### Signal Generation (`signals/`)
- **SignalGenerator**: Detects ICT patterns (FVG, Order Blocks, etc.)
- **ConfluenceValidator**: Validates signals across multiple conditions
- **SignalStrengthCalculator**: Quantifies signal quality

#### Strategy Layer (`strategies/`)
- **StrategySelector**: Dynamic strategy selection based on conditions
- **ICTStrategy**: Implements Inner Circle Trader methodology
- **BreakoutStrategy**: Breakout-based trading logic

#### Execution Layer (`execution/`)
- **OrderExecutor**: Places and manages orders
- **PositionManager**: Tracks open positions and P&L
- Handles partial fills, order amendments, cancellations

#### Risk Management (`risk/`)
- **RiskManager**: Per-trade risk validation
- **PortfolioRisk**: Portfolio-level risk aggregation
- Position sizing, exposure limits, drawdown monitoring

#### Notifications (`notifications/`)
- **DiscordNotifier**: Real-time alerts via Discord
- Sends trade notifications, system status, errors

---

## Development Workflows

### Running Locally

Development mode with live updates and detailed logging:

```bash
./scripts/run_local.sh
```

**Features**:
- Uses `development` environment configuration
- Connects to Binance testnet (if configured)
- Detailed console logging
- Small position sizes for safety
- Auto-reloads on configuration changes

**Configuration**: `config/environments/development.yml`

### Paper Trading

Safe testing with simulated funds on testnet:

```bash
./scripts/run_paper.sh
```

**Features**:
- Uses `paper` environment configuration
- Connects to Binance testnet API
- Simulated order execution with realistic slippage
- Virtual capital: 25,000 USDT
- No real money at risk

**Setup**:
1. Get testnet API keys: https://testnet.binance.vision/
2. Add to `.env`:
   ```bash
   TRADING_BINANCE_API_KEY=testnet_key
   TRADING_BINANCE_API_SECRET=testnet_secret
   ```

### Backtesting

Test strategies against historical data:

```bash
# Backtest default symbol (BTCUSDT)
./scripts/run_backtest.sh --start 2024-01-01 --end 2024-12-31

# Backtest specific symbol
./scripts/run_backtest.sh --symbol ETHUSDT --start 2024-06-01 --end 2024-09-30

# Verbose backtest with custom config
./scripts/run_backtest.sh --start 2024-01-01 --end 2024-12-31 --verbose --config custom.yaml
```

**Features**:
- Downloads historical data if not cached
- Simulates realistic order execution
- Generates performance reports
- Calculates risk-adjusted metrics (Sharpe, Sortino, max drawdown)

**Output**: Results saved to `results/backtests/`

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading_bot --cov-report=html

# Run specific test file
pytest tests/unit/test_signal_generator.py

# Run with verbose output
pytest -v

# Run integration tests only
pytest tests/integration/

# Run tests matching pattern
pytest -k "test_order_execution"
```

### Development Iteration

Typical development cycle:

1. **Make code changes**
2. **Run unit tests**:
   ```bash
   pytest tests/unit/ -v
   ```
3. **Run integration tests**:
   ```bash
   pytest tests/integration/ -v
   ```
4. **Test locally**:
   ```bash
   ./scripts/run_local.sh
   ```
5. **Paper trade validation**:
   ```bash
   ./scripts/run_paper.sh
   ```
6. **Backtest validation**:
   ```bash
   ./scripts/run_backtest.sh --start 2024-01-01 --end 2024-12-31
   ```

---

## Configuration Guide

### Environment Files

Configuration is environment-based using YAML files and `.env` for secrets.

#### `.env` File

Contains sensitive credentials (never commit to version control):

```bash
# API Credentials
TRADING_BINANCE_API_KEY=your_api_key_here
TRADING_BINANCE_API_SECRET=your_api_secret_here

# Discord Notifications (optional)
TRADING_DISCORD_BOT_TOKEN=your_discord_bot_token
TRADING_DISCORD_CHANNEL_ID=your_channel_id

# Environment Selection
TRADING_ENV=development  # development | testing | paper | production

# Optional Overrides
# TRADING_CAPITAL=10000.0
# TRADING_LEVERAGE=2
# TRADING_LOG_LEVEL=INFO
```

#### Environment Configurations

Located in `config/environments/`:

**development.yml** - Local development:
```yaml
api:
  testnet: true  # Use testnet API
  rate_limit_per_minute: 1200

risk:
  max_position_size: 1000.0  # Small for safety
  max_portfolio_risk: 0.02   # 2% of capital
  risk_per_trade: 0.01       # 1% per trade

logging:
  level: DEBUG  # Detailed logs
  console: true
```

**paper.yml** - Paper trading (testnet):
```yaml
api:
  testnet: true
  rate_limit_per_minute: 1200

risk:
  max_position_size: 5000.0
  max_portfolio_risk: 0.05
  risk_per_trade: 0.01

execution:
  simulate_slippage: true  # Realistic simulation
  simulate_fees: true
```

**production.yml** - Live trading (use with extreme caution):
```yaml
api:
  testnet: false  # REAL MONEY
  rate_limit_per_minute: 1200

risk:
  max_position_size: 10000.0
  max_portfolio_risk: 0.03
  risk_per_trade: 0.01

logging:
  level: INFO
  mask_sensitive: true  # Hide API keys in logs
```

### Configuration Structure

**Full configuration example** (`config/trading_config.yaml`):

```yaml
# Capital and leverage
capital: 50000.0
leverage: 3

# Market data
symbols:
  - BTCUSDT
  - ETHUSDT

data:
  cache_size: 1000
  cache_ttl: 300  # seconds
  historical_lookback_days: 90

# Risk management
risk:
  max_position_size: 10000.0
  max_portfolio_risk: 0.03  # 3% max portfolio risk
  risk_per_trade: 0.01      # 1% per trade
  max_drawdown: 0.15        # 15% max drawdown

# Strategy configuration
strategy:
  default_timeframes: ["15m", "1h", "4h"]
  confirmation_required: true
  min_signal_strength: 0.6

# Execution
execution:
  order_type: "limit"  # limit | market
  slippage_tolerance: 0.001  # 0.1%
  retry_attempts: 3
  timeout: 30  # seconds

# Notifications
notifications:
  enabled: true
  channels: ["discord"]
  notify_on_signal: true
  notify_on_entry: true
  notify_on_exit: true
  notify_on_error: true

# Logging
logging:
  level: INFO  # DEBUG | INFO | WARNING | ERROR
  console: true
  file: true
  max_file_size: 10485760  # 10MB
  backup_count: 5
```

### Environment Variable Substitution

Use `${VAR}` syntax in YAML to reference environment variables:

```yaml
api:
  api_key: ${TRADING_BINANCE_API_KEY}
  api_secret: ${TRADING_BINANCE_API_SECRET}

notifications:
  discord:
    bot_token: ${TRADING_DISCORD_BOT_TOKEN}
    channel_id: ${TRADING_DISCORD_CHANNEL_ID}
```

### Adding New Strategies

1. **Create strategy file** in `src/trading_bot/strategies/`:
   ```python
   # src/trading_bot/strategies/my_strategy.py
   from trading_bot.core.base_component import BaseComponent
   
   class MyStrategy(BaseComponent):
       async def analyze(self, market_data):
           # Strategy logic
           pass
   ```

2. **Register in SystemIntegrator**:
   ```python
   # In src/trading_bot/system_integrator.py
   from trading_bot.strategies.my_strategy import MyStrategy
   
   # In _register_components():
   my_strategy = MyStrategy(self.config, self.event_bus)
   await my_strategy.start()
   ```

3. **Add configuration**:
   ```yaml
   # In config/trading_config.yaml
   strategy:
     my_strategy:
       enabled: true
       parameter1: value1
   ```

### Adding New Indicators

1. **Create indicator** in `src/trading_bot/indicators/`:
   ```python
   def my_indicator(data: pd.DataFrame, period: int = 14) -> pd.Series:
       # Indicator calculation
       return result
   ```

2. **Use in strategy**:
   ```python
   from trading_bot.indicators.my_indicator import my_indicator
   
   result = my_indicator(market_data, period=20)
   ```

---

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests (isolated)
│   ├── test_signal_generator.py
│   ├── test_risk_manager.py
│   └── test_order_executor.py
├── integration/            # Integration tests (components)
│   ├── test_strategy_integration.py
│   ├── test_data_pipeline.py
│   └── test_system_integration.py
└── data/                   # Test fixtures
    ├── sample_market_data.csv
    └── mock_responses.json
```

### Writing Tests

**Unit Test Example**:
```python
# tests/unit/test_signal_generator.py
import pytest
from trading_bot.signals.signal_generator import SignalGenerator

@pytest.fixture
def signal_generator():
    config = {"timeframes": ["15m", "1h"]}
    return SignalGenerator(config)

def test_fvg_detection(signal_generator):
    # Arrange
    market_data = create_test_data_with_fvg()
    
    # Act
    signals = signal_generator.detect_fvg(market_data)
    
    # Assert
    assert len(signals) > 0
    assert signals[0].type == "FVG"
    assert signals[0].confidence > 0.7
```

**Integration Test Example**:
```python
# tests/integration/test_strategy_integration.py
import pytest
from trading_bot.strategies.ict_strategy import ICTStrategy

@pytest.mark.asyncio
async def test_end_to_end_signal_to_order():
    # Arrange
    strategy = ICTStrategy(config, event_bus)
    await strategy.start()
    
    # Act
    market_data = await fetch_test_market_data()
    signals = await strategy.analyze(market_data)
    
    # Assert
    assert len(signals) > 0
    # Verify order was created
    orders = await get_pending_orders()
    assert len(orders) > 0
```

### Test Coverage

Aim for >80% coverage on critical paths:

```bash
# Generate coverage report
pytest --cov=trading_bot --cov-report=html

# Open in browser
open htmlcov/index.html
```

### CI/CD Testing

Tests run automatically on:
- Every commit to feature branches
- Pull requests to main
- Before deployments

Ensure all tests pass before merging:
```bash
pytest -v --cov=trading_bot --cov-report=term-missing
```

---

## Troubleshooting

### Common Issues

#### 1. API Connection Failures

**Problem**: `ConnectionError: Unable to connect to Binance API`

**Solutions**:
- Check internet connection
- Verify API credentials in `.env`
- Ensure using testnet keys for development/paper trading
- Check API key permissions (needs trading enabled)
- Verify rate limits not exceeded

**Debug**:
```bash
# Test API connection
python -c "from trading_bot.data.binance_client import BinanceClient; client = BinanceClient(config); print(client.test_connection())"
```

#### 2. Configuration Errors

**Problem**: `ConfigurationError: Missing required field`

**Solutions**:
- Verify `.env` file exists and is properly formatted
- Check YAML syntax in environment config files
- Ensure all required environment variables are set
- Validate configuration with schema

**Debug**:
```bash
# Validate configuration
python -m trading_bot.config.config_manager --validate
```

#### 3. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'trading_bot'`

**Solutions**:
- Install package in development mode: `pip install -e .`
- Activate virtual environment: `source venv/bin/activate`
- Check Python version: `python --version` (needs 3.10+)
- Verify PYTHONPATH includes project root

#### 4. Dependency Issues

**Problem**: `ImportError: cannot import name 'X' from 'Y'`

**Solutions**:
- Update dependencies: `pip install --upgrade -r requirements.txt`
- Clear pip cache: `pip cache purge`
- Reinstall package: `pip uninstall trading_bot && pip install -e .`
- Check for version conflicts: `pip list --outdated`

#### 5. Performance Issues

**Problem**: High CPU usage or slow response times

**Solutions**:
- Check WebSocket connection stability
- Reduce number of symbols being monitored
- Increase cache TTL in configuration
- Review log level (DEBUG is verbose, use INFO in production)
- Monitor event queue sizes

**Debug**:
```bash
# Check system resource usage
python -m trading_bot.diagnostics.performance_monitor
```

#### 6. Database Lock Errors (SQLite)

**Problem**: `sqlite3.OperationalError: database is locked`

**Solutions**:
- Close other connections to database
- Increase timeout in configuration
- Consider PostgreSQL for production
- Ensure proper connection cleanup

### Logging and Debugging

**Enable verbose logging**:
```bash
# Set in .env
TRADING_LOG_LEVEL=DEBUG

# Or run with verbose flag
./scripts/run_local.sh --verbose
```

**Log locations**:
- Console output: stdout/stderr
- File logs: `logs/trading_bot.log`
- Error logs: `logs/error.log`
- Backtest logs: `logs/backtest/`

**Debug specific component**:
```python
import logging
logging.getLogger('trading_bot.signals').setLevel(logging.DEBUG)
```

### Getting Help

1. **Check documentation**: Review this guide and inline code comments
2. **Review logs**: Check `logs/` directory for error details
3. **Run diagnostics**: `python -m trading_bot.diagnostics`
4. **Search issues**: Check GitHub issues for similar problems
5. **Ask for help**: Create issue with logs, configuration, and error details

---

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make changes with tests
5. Run quality checks:
   ```bash
   # Format code
   black src/ tests/
   
   # Lint
   flake8 src/ tests/
   
   # Type check
   mypy src/
   
   # Run tests
   pytest -v --cov=trading_bot
   ```
6. Commit with clear message
7. Push and create pull request

### Code Standards

- **Style**: Follow PEP 8, use Black formatter
- **Type hints**: Required for all functions
- **Documentation**: Docstrings for public APIs
- **Tests**: >80% coverage for new code
- **Commits**: Clear, descriptive messages

### Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address review feedback
6. Maintainer will merge when approved

---

## Additional Resources

- [ICT Trading Methodology](https://www.innercircletrader.com/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/spot/en/)
- [Python Async Programming](https://docs.python.org/3/library/asyncio.html)
- [Pydantic Configuration](https://docs.pydantic.dev/)
- [Pytest Documentation](https://docs.pytest.org/)

---

**Last Updated**: 2025-10-02

**Version**: 1.0.0

**Maintained By**: Trading Bot Development Team
