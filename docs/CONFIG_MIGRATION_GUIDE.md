# Configuration Migration Guide

## Overview

This guide explains how to migrate from the legacy `trading_config.yaml` format to the new environment-based configuration system.

### Why Migrate?

The new configuration system provides:

- **Environment Separation**: Dedicated config files for development, paper trading, and production
- **Better Organization**: Clearer separation of concerns with dedicated sections
- **Type Safety**: Pydantic v2 validation ensures configuration correctness
- **Network Awareness**: Explicit testnet/mainnet and paper/live mode configuration
- **Improved Security**: Better handling of sensitive data and environment variables
- **Reduced Duplication**: Shared base configuration with environment-specific overrides

### What Changed?

#### Old Structure (Single File)
```
config/
└── trading_config.yaml    # All environments in one file
```

#### New Structure (Environment-Based)
```
src/trading_bot/config/
└── environments/
    ├── base.yml              # Shared settings
    ├── development.yml       # Local development
    ├── paper-testnet.yml     # Paper trading on testnet
    ├── paper-mainnet.yml     # Paper trading on mainnet
    ├── paper.yml             # Alias for paper-testnet
    ├── testing.yml           # Automated testing
    └── production.yml        # Live trading
```

## Migration Process

### Prerequisites

Before starting migration:

1. **Backup your current configuration**
   ```bash
   cp config/trading_config.yaml config/trading_config.yaml.backup
   ```

2. **Ensure Python dependencies are installed**
   ```bash
   pip install -e .
   ```

3. **Verify you have PyYAML**
   ```bash
   pip install pyyaml
   ```

### Step 1: Run Migration Script

The migration script automatically converts your old configuration to the new format.

```bash
python scripts/config_migration.py config/trading_config.yaml
```

**Options:**
- `--output-dir <dir>`: Specify output directory (default: `src/trading_bot/config/environments`)
- `--environments <env1,env2>`: Generate specific environments only

**Example:**
```bash
# Generate only development and production
python scripts/config_migration.py config/trading_config.yaml \
    --environments development,production
```

### Step 2: Review Generated Files

After migration, review the generated environment files:

```bash
ls -la src/trading_bot/config/environments/
```

You should see files like:
- `development.yml`
- `paper-testnet.yml`
- `paper-mainnet.yml`
- `production.yml`

### Step 3: Validate Configuration

Use the validation tool to ensure configurations are correct:

```bash
# Validate all environments
python scripts/validate_config.py

# Validate specific environment
python scripts/validate_config.py --environment development
```

The validator will:
- ✅ Check Pydantic model compliance
- ⚠️  Warn about required environment variables
- ⚠️  Flag potential security issues (e.g., DEBUG in production)
- ⚠️  Highlight high leverage settings

### Step 4: Set Environment Variables

Create or update your `.env` file with required variables:

```bash
# Required for all environments
TRADING_BINANCE_API_KEY=your_api_key_here
TRADING_BINANCE_API_SECRET=your_api_secret_here

# Optional: Discord notifications
TRADING_DISCORD_BOT_TOKEN=your_bot_token
TRADING_DISCORD_CHANNEL_ID=your_channel_id

# Development/Testing (use testnet keys)
# TRADING_BINANCE_API_KEY=testnet_api_key
# TRADING_BINANCE_API_SECRET=testnet_api_secret
```

### Step 5: Test Configuration

Test the new configuration with your chosen environment:

```bash
# Test development environment
TRADING_ENV=development python -m trading_bot.main --validate-config

# Test paper trading environment
TRADING_ENV=paper-testnet python -m trading_bot.main --validate-config

# Test production environment (be careful!)
TRADING_ENV=production python -m trading_bot.main --validate-config
```

### Step 6: Compare Environments (Optional)

Use the comparison tool to understand differences across environments:

```bash
# Compare all environments
python scripts/compare_environments.py

# Compare specific environments
python scripts/compare_environments.py --environments development,production

# Export comparison to JSON
python scripts/compare_environments.py --export comparison_report.json
```

## Configuration Mapping

### Top-Level Fields

| Old Field | New Location | Notes |
|-----------|--------------|-------|
| `capital` | `risk.initial_capital` | Moved to risk management section |
| `leverage` | `trading.leverage` | Now environment-specific |
| `symbols` | `trading.symbol` | Changed to single symbol (pick first) |

### Section Mappings

#### Data Configuration
| Old Field | New Field | Changes |
|-----------|-----------|---------|
| `data.cache_size` | `data.cache_size` | No change |
| `data.cache_ttl` | `data.cache_ttl` | No change |
| `data.historical_lookback_days` | `data.historical_lookback_days` | No change |

#### Risk Management
| Old Field | New Field | Changes |
|-----------|-----------|---------|
| `risk.max_position_size` | `risk.initial_capital` | Renamed for clarity |
| `risk.risk_per_trade` | `risk.max_position_size_pct` | Renamed to reflect percentage |
| `risk.max_drawdown` | `risk.max_drawdown_pct` | Renamed to reflect percentage |
| N/A | `risk.max_consecutive_losses` | New field (default: 3) |
| N/A | `risk.volatility_threshold` | New field (default: 0.05) |

#### Execution
| Old Field | New Field | Changes |
|-----------|-----------|---------|
| `execution.slippage_tolerance` | `execution.slippage_tolerance_pct` | Renamed for clarity |
| `execution.retry_attempts` | `execution.max_retry_attempts` | Renamed for clarity |
| `execution.timeout` | `execution.order_timeout_seconds` | Renamed for clarity |

#### Logging
| Old Field | New Field | Changes |
|-----------|-----------|---------|
| `logging.level` | `logging.level` | No change |
| `logging.max_file_size` | `logging.max_bytes` | Renamed to match Python logging |
| `logging.backup_count` | `logging.backup_count` | No change |
| N/A | `logging.format` | New field (default: "json") |
| N/A | `logging.mask_sensitive_data` | New security feature |

#### Notifications → Discord
| Old Field | New Field | Changes |
|-----------|-----------|---------|
| `notifications.enabled` | `discord.enable_notifications` | Moved to discord section |
| `notifications.discord.bot_token` | `discord.bot_token` | Flattened structure |
| `notifications.discord.channel_id` | `discord.channel_id` | Flattened structure |

### New Sections

The new configuration adds these sections:

#### Binance Configuration
```yaml
binance:
  api_key: ${TRADING_BINANCE_API_KEY}
  api_secret: ${TRADING_BINANCE_API_SECRET}
  network: testnet              # testnet | mainnet
  execution_mode: paper         # paper | live
```

#### System Configuration
```yaml
system:
  async_workers: 2
  event_queue_size: 1000
  health_check_interval: 300
  memory_threshold_mb: 1024
```

## Environment-Specific Settings

### Development
- **Network**: testnet
- **Execution Mode**: paper
- **Logging**: DEBUG level
- **Leverage**: 1x (conservative)

### Paper Trading (Testnet)
- **Network**: testnet
- **Execution Mode**: paper
- **Logging**: INFO level
- **Leverage**: 1x (conservative)

### Paper Trading (Mainnet)
- **Network**: mainnet
- **Execution Mode**: paper
- **Logging**: INFO level
- **Leverage**: 2x (moderate)

### Production
- **Network**: mainnet
- **Execution Mode**: live
- **Logging**: INFO level, sensitive data masked
- **Leverage**: From original config or 3x default

## Troubleshooting

### Migration Script Errors

**Error: "Input file not found"**
```bash
# Check file path is correct
ls -la config/trading_config.yaml

# Try with absolute path
python scripts/config_migration.py /full/path/to/trading_config.yaml
```

**Error: "YAML parsing error"**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/trading_config.yaml'))"

# Check for tabs vs spaces (YAML requires spaces)
cat -A config/trading_config.yaml | grep $'\t'
```

### Validation Errors

**Error: "Missing required field"**
- Check that all required fields are present in environment file
- Ensure base.yml is in the environments directory
- Verify field names match Pydantic models

**Error: "Environment variable not set"**
```bash
# List required environment variables
python scripts/validate_config.py --environment production

# Set in .env file
echo "TRADING_BINANCE_API_KEY=your_key" >> .env
```

### Configuration Doesn't Load

**Issue: Bot uses wrong environment**
```bash
# Explicitly set environment
export TRADING_ENV=development
python -m trading_bot.main

# Or set in command
TRADING_ENV=production python -m trading_bot.main
```

**Issue: Environment variables not expanded**
```bash
# Check .env file is in project root
ls -la .env

# Load .env in shell
export $(cat .env | xargs)

# Or use python-dotenv
pip install python-dotenv
```

## Verification Checklist

After migration, verify:

- [ ] All environment files are generated
- [ ] Validation passes for all environments
- [ ] Required environment variables are set in `.env`
- [ ] Network settings are correct (testnet for dev/paper, mainnet for production)
- [ ] Execution mode is correct (paper for testing, live for production)
- [ ] Leverage settings are appropriate for each environment
- [ ] Logging level is appropriate (DEBUG for dev, INFO for production)
- [ ] Sensitive data masking is enabled in production
- [ ] Old `config/trading_config.yaml` is backed up or removed

## Best Practices

### Environment Variable Management

1. **Never commit `.env` to version control**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use `.env.example` for documentation**
   ```bash
   # Create template
   cat > .env.example << EOF
   TRADING_BINANCE_API_KEY=your_api_key_here
   TRADING_BINANCE_API_SECRET=your_api_secret_here
   TRADING_DISCORD_BOT_TOKEN=optional_discord_bot_token
   TRADING_DISCORD_CHANNEL_ID=optional_discord_channel_id
   EOF
   ```

3. **Use different API keys per environment**
   - Development: Testnet keys with low limits
   - Paper Trading: Testnet keys for simulation
   - Production: Mainnet keys with proper security

### Configuration Security

1. **Sensitive data in environment variables only**
   - Never hardcode API keys in YAML files
   - Use `${VARIABLE}` syntax for all secrets

2. **Enable sensitive data masking**
   ```yaml
   logging:
     mask_sensitive_data: true
     debug_unmask: false
   ```

3. **Use appropriate logging levels**
   - Development: DEBUG (detailed)
   - Paper Trading: INFO (moderate)
   - Production: INFO with masking

### Testing Strategy

1. **Always test in paper trading first**
   ```bash
   TRADING_ENV=paper-testnet python -m trading_bot.main
   ```

2. **Validate before deploying**
   ```bash
   python scripts/validate_config.py --environment production
   ```

3. **Compare configurations regularly**
   ```bash
   python scripts/compare_environments.py
   ```

## Migration Tools Reference

### config_migration.py

**Purpose**: Convert legacy config to new environment-based structure

**Usage**:
```bash
python scripts/config_migration.py <input_file> [options]
```

**Options**:
- `--output-dir <dir>`: Output directory for environment files
- `--environments <list>`: Comma-separated environments to generate

**Example**:
```bash
python scripts/config_migration.py config/trading_config.yaml \
    --output-dir src/trading_bot/config/environments \
    --environments development,paper-testnet,production
```

### validate_config.py

**Purpose**: Validate configuration files against Pydantic models

**Usage**:
```bash
python scripts/validate_config.py [options]
```

**Options**:
- `--environment <env>`: Validate specific environment
- `--config-dir <dir>`: Configuration directory
- `--export <file>`: Export validation results to JSON

**Example**:
```bash
# Validate all
python scripts/validate_config.py

# Validate specific environment
python scripts/validate_config.py --environment production

# Export results
python scripts/validate_config.py --export validation_report.json
```

### compare_environments.py

**Purpose**: Compare settings across environments

**Usage**:
```bash
python scripts/compare_environments.py [options]
```

**Options**:
- `--environments <list>`: Environments to compare
- `--config-dir <dir>`: Configuration directory
- `--show-all`: Show all settings, not just differences
- `--export <file>`: Export comparison to JSON

**Example**:
```bash
# Compare all
python scripts/compare_environments.py

# Compare specific
python scripts/compare_environments.py --environments development,production

# Show all settings
python scripts/compare_environments.py --show-all

# Export results
python scripts/compare_environments.py --export comparison.json
```

## Additional Resources

- [Development Guide](development.md) - General development documentation
- [Configuration Models](../src/trading_bot/config/models.py) - Pydantic validation models
- [Config Manager](../src/trading_bot/config/config_manager.py) - Configuration loading logic
- [Environment Variables Guide](ENVIRONMENT_VARIABLES.md) - Complete environment variable reference

## Support

If you encounter issues during migration:

1. Check this guide's troubleshooting section
2. Validate configuration files with the validation tool
3. Review validation error messages carefully
4. Ensure all environment variables are set correctly
5. Test in paper trading before production

For additional help, create an issue with:
- Migration script output
- Validation errors (if any)
- Environment configuration files
- Error messages and stack traces
