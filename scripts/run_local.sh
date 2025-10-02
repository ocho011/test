#!/bin/bash

# Trading Bot - Local Development Runner
# Starts the trading bot in development mode with proper validation

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

Run the trading bot in local development mode.

OPTIONS:
    -h, --help          Show this help message
    -c, --config PATH   Use custom configuration file
    -v, --verbose       Enable verbose logging

EXAMPLES:
    $0                              # Run with default settings
    $0 --config custom_config.yaml  # Run with custom configuration
    $0 --verbose                    # Run with verbose logging

ENVIRONMENT:
    Requires .env file with:
        - TRADING_BINANCE_API_KEY
        - TRADING_BINANCE_API_SECRET
        - Other optional configuration variables

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
        print_info "  # Then edit .env with your API credentials"
        exit 1
    fi
    
    # Check if running in virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "Not running in a virtual environment"
        print_info "Recommended: Activate virtual environment first"
        print_info "  python -m venv venv"
        print_info "  source venv/bin/activate  # On Unix/macOS"
        print_info "  venv\\Scripts\\activate     # On Windows"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    REQUIRED_VERSION="3.10.0"
    if ! python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        print_error "Python 3.10+ required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    
    # Check if required packages are installed
    if ! python -c "import trading_bot" 2>/dev/null; then
        print_error "trading_bot package not installed!"
        print_info "Please install dependencies:"
        print_info "  pip install -e ."
        exit 1
    fi
    
    print_success "Environment validation passed"
}

# Function to check configuration
function check_configuration() {
    print_info "Checking configuration..."
    
    # Source .env file to check required variables
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    
    if [ -z "$TRADING_BINANCE_API_KEY" ]; then
        print_error "TRADING_BINANCE_API_KEY not set in .env"
        exit 1
    fi
    
    if [ -z "$TRADING_BINANCE_API_SECRET" ]; then
        print_error "TRADING_BINANCE_API_SECRET not set in .env"
        exit 1
    fi
    
    print_success "Configuration validated"
}

# Function to run the bot
function run_bot() {
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
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    print_info "Starting trading bot in development mode..."
    print_info "Press Ctrl+C to stop gracefully"
    echo ""
    
    # Run the bot with development environment
    python -m trading_bot.main \
        --env development \
        --mode trading \
        $config_arg \
        $verbose_arg
}

# Main execution
function main() {
    echo "========================================="
    echo "  Trading Bot - Local Development"
    echo "========================================="
    echo ""
    
    validate_environment
    check_configuration
    
    echo ""
    run_bot "$@"
}

# Trap Ctrl+C for graceful shutdown message
trap 'echo ""; print_info "Shutdown signal received, bot stopping gracefully..."; exit 0' INT

main "$@"
