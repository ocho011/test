#!/bin/bash

# Trading Bot - Paper Trading Runner
# Runs the bot in paper trading mode using testnet

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
function print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

function print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
function show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run the trading bot in paper trading mode (testnet).

Paper trading allows you to test strategies with simulated funds on the
Binance testnet without risking real capital.

OPTIONS:
    -h, --help          Show this help message
    -c, --config PATH   Use custom configuration file
    -v, --verbose       Enable verbose logging

EXAMPLES:
    $0                              # Run with default paper config
    $0 --config custom_paper.yaml   # Run with custom configuration
    $0 --verbose                    # Run with verbose logging

REQUIREMENTS:
    - Binance testnet API credentials in .env file
    - TRADING_BINANCE_API_KEY (testnet key)
    - TRADING_BINANCE_API_SECRET (testnet secret)

TESTNET SETUP:
    1. Get testnet API keys from: https://testnet.binance.vision/
    2. Add to .env file:
       TRADING_BINANCE_API_KEY=your_testnet_api_key
       TRADING_BINANCE_API_SECRET=your_testnet_api_secret
    3. Ensure TRADING_ENV=paper or use paper environment config

FEATURES:
    ✓ Real-time market data from Binance testnet
    ✓ Simulated order execution with realistic slippage
    ✓ Simulated fees matching production
    ✓ Virtual capital: 25,000 USDT (configurable)
    ✓ Full strategy testing without financial risk
    ✓ Live monitoring and notifications

SAFETY:
    - No real capital at risk
    - Testnet only, no production API access
    - Conservative position sizing
    - All safety features enabled

EOF
}

# Function to validate environment
function validate_environment() {
    print_info "Validating environment..."
    
    # Check if .env file exists
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        print_error ".env file not found!"
        print_info "Please create .env file from .env.template:"
        print_info "  cp .env.template .env"
        print_info "  # Then add your testnet API credentials"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    if ! python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        print_error "Python 3.10+ required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    
    # Check if trading_bot package is installed
    if ! python -c "import trading_bot" 2>/dev/null; then
        print_error "trading_bot package not installed!"
        print_info "Please install: pip install -e ."
        exit 1
    fi
    
    print_success "Environment validation passed"
}

# Function to check testnet configuration
function check_testnet_config() {
    print_info "Checking testnet configuration..."
    
    # Source .env file
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    
    if [ -z "$TRADING_BINANCE_API_KEY" ]; then
        print_error "TRADING_BINANCE_API_KEY not set in .env"
        print_info "Get testnet API key from: https://testnet.binance.vision/"
        exit 1
    fi
    
    if [ -z "$TRADING_BINANCE_API_SECRET" ]; then
        print_error "TRADING_BINANCE_API_SECRET not set in .env"
        print_info "Get testnet API secret from: https://testnet.binance.vision/"
        exit 1
    fi
    
    # Warn if using production environment setting
    if [ "$TRADING_ENV" = "production" ]; then
        print_warning "TRADING_ENV is set to 'production'"
        print_warning "Paper trading will override this to use 'paper' environment"
    fi
    
    print_success "Testnet configuration validated"
}

# Function to show safety warning
function show_safety_warning() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              PAPER TRADING MODE - TESTNET                  ║"
    echo "╟────────────────────────────────────────────────────────────╢"
    echo "║  ✓ Using Binance testnet - NO REAL MONEY AT RISK         ║"
    echo "║  ✓ Simulated trading with virtual capital                 ║"
    echo "║  ✓ Real-time market data for realistic testing            ║"
    echo "║  ✓ All strategies and risk management active              ║"
    echo "║                                                            ║"
    echo "║  Virtual Capital: 25,000 USDT                             ║"
    echo "║  Max Position Size: 5,000 USDT                            ║"
    echo "║  Risk per Trade: 1% of capital                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    print_info "Press Ctrl+C to stop the bot gracefully at any time"
    sleep 2
}

# Function to run paper trading
function run_paper_trading() {
    local config_arg=""
    local verbose_arg=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                config_arg="--config $2"
                shift 2
                ;;
            -v|--verbose)
                verbose_arg="--verbose"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    cd "$PROJECT_ROOT"
    
    print_info "Starting paper trading mode..."
    echo ""
    
    # Run the bot with paper environment
    python -m trading_bot.main \
        --env paper \
        --mode paper \
        $config_arg \
        $verbose_arg
}

# Main execution
function main() {
    echo "========================================="
    echo "  Trading Bot - Paper Trading"
    echo "========================================="
    echo ""
    
    validate_environment
    check_testnet_config
    show_safety_warning
    
    run_paper_trading "$@"
}

# Trap Ctrl+C for graceful shutdown
trap 'echo ""; print_info "Shutdown signal received, stopping paper trading..."; exit 0' INT

main "$@"
