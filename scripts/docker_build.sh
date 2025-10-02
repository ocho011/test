#!/bin/bash
# Docker build and run helper script for ICT Trading Bot

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Function to build Docker image
build_image() {
    local env=${1:-dev}
    print_info "Building Docker image for $env environment..."
    
    if [ "$env" = "dev" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
    elif [ "$env" = "prod" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml build
    else
        docker-compose build
    fi
    
    print_info "Build completed successfully!"
}

# Function to start containers
start_containers() {
    local env=${1:-dev}
    print_info "Starting containers for $env environment..."
    
    if [ "$env" = "dev" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    elif [ "$env" = "prod" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        docker-compose up -d
    fi
    
    print_info "Containers started successfully!"
    print_info "View logs with: docker-compose logs -f trading-bot"
}

# Function to stop containers
stop_containers() {
    print_info "Stopping containers..."
    docker-compose down
    print_info "Containers stopped successfully!"
}

# Function to view logs
view_logs() {
    print_info "Viewing logs (Ctrl+C to exit)..."
    docker-compose logs -f trading-bot
}

# Function to run tests in container
run_tests() {
    print_info "Running tests in container..."
    docker-compose run --rm trading-bot python -m pytest tests/ -v
}

# Function to clean up
cleanup() {
    print_warn "This will remove all containers, volumes, and images. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "Cleaning up Docker resources..."
        docker-compose down -v --rmi all
        print_info "Cleanup completed!"
    else
        print_info "Cleanup cancelled."
    fi
}

# Main script logic
case "${1:-help}" in
    build)
        build_image "${2:-dev}"
        ;;
    start)
        start_containers "${2:-dev}"
        ;;
    stop)
        stop_containers
        ;;
    restart)
        stop_containers
        start_containers "${2:-dev}"
        ;;
    logs)
        view_logs
        ;;
    test)
        run_tests
        ;;
    clean)
        cleanup
        ;;
    *)
        echo "Usage: $0 {build|start|stop|restart|logs|test|clean} [dev|prod]"
        echo ""
        echo "Commands:"
        echo "  build [env]    Build Docker image (default: dev)"
        echo "  start [env]    Start containers (default: dev)"
        echo "  stop           Stop containers"
        echo "  restart [env]  Restart containers (default: dev)"
        echo "  logs           View container logs"
        echo "  test           Run tests in container"
        echo "  clean          Remove all Docker resources"
        echo ""
        echo "Examples:"
        echo "  $0 build dev       # Build development image"
        echo "  $0 start prod      # Start production containers"
        echo "  $0 logs            # View logs"
        exit 1
        ;;
esac
