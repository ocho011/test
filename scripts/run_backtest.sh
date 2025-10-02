#!/bin/bash

# Trading Bot - Backtesting Runner
# Executes backtesting with historical data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SYMBOL="BTCUSDT"
START_DATE=""
END_DATE=""
CONFIG=""
VERBOSE=false

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

Run backtesting for trading strategies.

OPTIONS:
    -h, --help              Show this help message
    -s, --symbol SYMBOL     Trading pair (default: BTCUSDT)
    --start DATE            Start date (YYYY-MM-DD format, required)
    --end DATE              End date (YYYY-MM-DD format, required)
    -c, --config PATH       Use custom configuration file
    -v, --verbose           Enable verbose logging

EXAMPLES:
    # Backtest BTCUSDT for Q4 2024
    $0 --start 2024-10-01 --end 2024-12-31

    # Backtest ETHUSDT with custom config
    $0 --symbol ETHUSDT --start 2024-01-01 --end 2024-12-31 --config custom.yaml

    # Verbose backtest for analysis
    $0 --start 2024-06-01 --end 2024-09-30 --verbose

REQUIREMENTS:
    - Start and end dates are required
    - Dates must be in YYYY-MM-DD format
    - Start date must be before end date
    - Historical data will be downloaded if not cached

OUTPUT:
    - Backtest results will be saved to: results/backtests/
    - Detailed logs in: logs/backtest/
    - Performance metrics and trade history included

EOF
}

# Function to validate date format
function validate_date() {
    local date_str=$1
    local date_name=$2
    
    if [[ ! $date_str =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        print_error "$date_name must be in YYYY-MM-DD format"
        exit 1
    fi
    
    # Check if date is valid using Python
    if ! python -c "from datetime import datetime; datetime.strptime('$date_str', '%Y-%m-%d')" 2>/dev/null; then
        print_error "$date_name is not a valid date: $date_str"
        exit 1
    fi
}

# Function to validate date range
function validate_date_range() {
    if [ -z "$START_DATE" ]; then
        print_error "Start date is required (--start YYYY-MM-DD)"
        show_help
        exit 1
    fi
    
    if [ -z "$END_DATE" ]; then
        print_error "End date is required (--end YYYY-MM-DD)"
        show_help
        exit 1
    fi
    
    validate_date "$START_DATE" "Start date"
    validate_date "$END_DATE" "End date"
    
    # Check if start is before end using Python
    if ! python -c "from datetime import datetime; exit(0 if datetime.strptime('$START_DATE', '%Y-%m-%d') < datetime.strptime('$END_DATE', '%Y-%m-%d') else 1)"; then
        print_error "Start date must be before end date"
        exit 1
    fi
    
    print_success "Date range validated: $START_DATE to $END_DATE"
}

# Function to validate environment
function validate_environment() {
    print_info "Validating environment..."
    
    # Check Python version
    if ! python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        print_error "Python 3.10+ required"
        exit 1
    fi
    
    # Check if trading_bot package is installed
    if ! python -c "import trading_bot" 2>/dev/null; then
        print_error "trading_bot package not installed!"
        print_info "Please install: pip install -e ."
        exit 1
    fi
    
    print_success "Environment validated"
}

# Function to run backtest
function run_backtest() {
    cd "$PROJECT_ROOT"
    
    # Build command
    local cmd="python -m trading_bot.main --env testing --mode backtest --symbol $SYMBOL --start $START_DATE --end $END_DATE"
    
    if [ -n "$CONFIG" ]; then
        cmd="$cmd --config $CONFIG"
    fi
    
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd --verbose"
    fi
    
    print_info "Running backtest..."
    print_info "Symbol: $SYMBOL"
    print_info "Period: $START_DATE to $END_DATE"
    echo ""
    
    # Execute backtest
    eval $cmd
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        print_success "Backtest completed successfully!"
        print_info "Results saved to: results/backtests/"
        print_info "Logs available in: logs/backtest/"
    else
        echo ""
        print_error "Backtest failed with exit code: $exit_code"
        exit $exit_code
    fi
}

# Parse command-line arguments
function parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--symbol)
                SYMBOL="$2"
                shift 2
                ;;
            --start)
                START_DATE="$2"
                shift 2
                ;;
            --end)
                END_DATE="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Main execution
function main() {
    echo "========================================="
    echo "  Trading Bot - Backtesting"
    echo "========================================="
    echo ""
    
    parse_args "$@"
    validate_environment
    validate_date_range
    
    echo ""
    run_backtest
}

main "$@"
