#!/bin/bash
# Vultr VM Automated Deployment Script for ICT Trading Bot
# This script automates the complete deployment process on Vultr VMs including:
# - Docker installation and configuration
# - Essential environment setup
# - SSH key-based security
# - Firewall and security group configuration
# - Deployment automation with rollback capability

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# Script version
SCRIPT_VERSION="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables (can be overridden via environment or config file)
DEPLOYMENT_USER="${DEPLOYMENT_USER:-tradingbot}"
APP_DIR="${APP_DIR:-/opt/trading-bot}"
BACKUP_DIR="${BACKUP_DIR:-/opt/trading-bot-backups}"
LOG_DIR="${LOG_DIR:-/var/log/trading-bot}"
CONFIG_FILE="${CONFIG_FILE:-.env.production}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa.pub}"
DOCKER_VERSION="${DOCKER_VERSION:-latest}"
REQUIRED_DISK_SPACE_GB="${REQUIRED_DISK_SPACE_GB:-10}"

# Deployment state file for rollback
STATE_FILE="/tmp/deployment_state.json"

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

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to log to file
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_DIR/deployment.log"
}

# Function to save deployment state
save_state() {
    local step="$1"
    local status="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    cat > "$STATE_FILE" <<EOF
{
    "step": "$step",
    "status": "$status",
    "timestamp": "$timestamp",
    "version": "$SCRIPT_VERSION"
}
EOF
}

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Function to check system requirements
check_requirements() {
    print_step "Checking system requirements..."
    
    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        print_error "Cannot determine OS. /etc/os-release not found."
        exit 1
    fi
    
    source /etc/os-release
    print_info "OS: $NAME $VERSION"
    
    # Check if Ubuntu/Debian
    if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
        print_warn "This script is tested on Ubuntu/Debian. Proceed with caution on $NAME."
    fi
    
    # Check disk space
    local available_space=$(df / | tail -1 | awk '{print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_gb -lt $REQUIRED_DISK_SPACE_GB ]]; then
        print_error "Insufficient disk space. Required: ${REQUIRED_DISK_SPACE_GB}GB, Available: ${available_gb}GB"
        exit 1
    fi
    
    print_info "Disk space check passed: ${available_gb}GB available"
    
    # Check memory
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_mem -lt 1 ]]; then
        print_warn "Low memory detected: ${total_mem}GB. Recommended: 2GB+"
    else
        print_info "Memory check passed: ${total_mem}GB available"
    fi
    
    save_state "requirements_check" "completed"
}

# Function to setup logging
setup_logging() {
    print_step "Setting up logging..."
    
    mkdir -p "$LOG_DIR"
    chmod 755 "$LOG_DIR"
    
    # Create log file
    touch "$LOG_DIR/deployment.log"
    chmod 644 "$LOG_DIR/deployment.log"
    
    log "Deployment started - Version $SCRIPT_VERSION"
    print_info "Logging configured at $LOG_DIR/deployment.log"
    
    save_state "logging_setup" "completed"
}

# Function to install Docker
install_docker() {
    print_step "Installing Docker..."
    
    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
        print_info "Docker already installed: $docker_version"
        log "Docker already installed: $docker_version"
        return 0
    fi
    
    # Update package index
    apt-get update -qq
    
    # Install prerequisites
    apt-get install -y -qq \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Set up Docker repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Start and enable Docker
    systemctl start docker
    systemctl enable docker
    
    # Verify installation
    if docker --version &> /dev/null; then
        local docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
        print_info "Docker installed successfully: $docker_version"
        log "Docker installed: $docker_version"
    else
        print_error "Docker installation failed"
        exit 1
    fi
    
    save_state "docker_install" "completed"
}

# Function to create deployment user
create_deployment_user() {
    print_step "Creating deployment user: $DEPLOYMENT_USER..."
    
    if id "$DEPLOYMENT_USER" &>/dev/null; then
        print_info "User $DEPLOYMENT_USER already exists"
        log "User $DEPLOYMENT_USER already exists"
    else
        useradd -m -s /bin/bash "$DEPLOYMENT_USER"
        print_info "User $DEPLOYMENT_USER created"
        log "User $DEPLOYMENT_USER created"
    fi
    
    # Add user to docker group
    usermod -aG docker "$DEPLOYMENT_USER"
    print_info "User $DEPLOYMENT_USER added to docker group"
    
    save_state "user_creation" "completed"
}

# Function to setup SSH key authentication
setup_ssh_keys() {
    print_step "Setting up SSH key authentication..."
    
    local user_home=$(eval echo "~$DEPLOYMENT_USER")
    local ssh_dir="$user_home/.ssh"
    local authorized_keys="$ssh_dir/authorized_keys"
    
    # Create .ssh directory
    mkdir -p "$ssh_dir"
    chmod 700 "$ssh_dir"
    
    # Setup authorized_keys
    if [[ -f "$SSH_KEY_PATH" ]]; then
        cat "$SSH_KEY_PATH" >> "$authorized_keys"
        print_info "SSH key from $SSH_KEY_PATH added to authorized_keys"
        log "SSH key added from $SSH_KEY_PATH"
    else
        print_warn "SSH key not found at $SSH_KEY_PATH. Skipping key setup."
        print_warn "Please configure SSH keys manually for secure access."
    fi
    
    # Set proper permissions
    chmod 600 "$authorized_keys" 2>/dev/null || true
    chown -R "$DEPLOYMENT_USER:$DEPLOYMENT_USER" "$ssh_dir"
    
    # Configure SSH daemon for key-only authentication
    if grep -q "^PasswordAuthentication" /etc/ssh/sshd_config; then
        sed -i 's/^PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
    else
        echo "PasswordAuthentication no" >> /etc/ssh/sshd_config
    fi
    
    if grep -q "^PubkeyAuthentication" /etc/ssh/sshd_config; then
        sed -i 's/^PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
    else
        echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config
    fi
    
    # Restart SSH service
    systemctl restart sshd || systemctl restart ssh
    print_info "SSH configured for key-based authentication only"
    log "SSH key authentication configured"
    
    save_state "ssh_setup" "completed"
}

# Function to configure firewall
configure_firewall() {
    print_step "Configuring firewall (UFW)..."
    
    # Install UFW if not present
    if ! command -v ufw &> /dev/null; then
        apt-get install -y -qq ufw
    fi
    
    # Reset UFW to default state
    ufw --force reset
    
    # Default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH (critical - do this first!)
    ufw allow 22/tcp comment 'SSH'
    
    # Allow HTTP/HTTPS for potential web interface
    ufw allow 80/tcp comment 'HTTP'
    ufw allow 443/tcp comment 'HTTPS'
    
    # Allow custom application port if specified
    if [[ -n "${APP_PORT}" ]]; then
        ufw allow "${APP_PORT}/tcp" comment 'Application Port'
        print_info "Opened application port: $APP_PORT"
    fi
    
    # Enable UFW
    ufw --force enable
    
    print_info "Firewall configured and enabled"
    log "Firewall configured with UFW"
    
    # Display firewall status
    ufw status numbered
    
    save_state "firewall_setup" "completed"
}

# Function to install essential tools
install_essential_tools() {
    print_step "Installing essential system tools..."
    
    apt-get update -qq
    apt-get install -y -qq \
        git \
        htop \
        vim \
        curl \
        wget \
        jq \
        net-tools \
        software-properties-common \
        python3-pip \
        build-essential
    
    print_info "Essential tools installed"
    log "Essential tools installed"
    
    save_state "tools_install" "completed"
}

# Function to create application directories
create_app_directories() {
    print_step "Creating application directories..."
    
    # Create main directories
    mkdir -p "$APP_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$APP_DIR/logs"
    mkdir -p "$APP_DIR/data"
    mkdir -p "$APP_DIR/config"
    
    # Set ownership
    chown -R "$DEPLOYMENT_USER:$DEPLOYMENT_USER" "$APP_DIR"
    chown -R "$DEPLOYMENT_USER:$DEPLOYMENT_USER" "$BACKUP_DIR"
    
    print_info "Application directories created at $APP_DIR"
    log "Application directories created"
    
    save_state "directories_created" "completed"
}

# Function to setup application configuration
setup_app_config() {
    print_step "Setting up application configuration..."
    
    local config_path="$APP_DIR/.env"
    
    if [[ -f "$CONFIG_FILE" ]]; then
        cp "$CONFIG_FILE" "$config_path"
        chmod 600 "$config_path"
        chown "$DEPLOYMENT_USER:$DEPLOYMENT_USER" "$config_path"
        print_info "Configuration copied from $CONFIG_FILE"
        log "Configuration file deployed"
    else
        print_warn "Configuration file not found: $CONFIG_FILE"
        print_warn "Creating template configuration file"
        
        cat > "$config_path" <<'EOF'
# Trading Bot Production Configuration
# Generated by deployment script

# Environment
TRADING_ENV=production
DEBUG=false

# API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Trading Parameters
INITIAL_CAPITAL=10000
RISK_PER_TRADE=0.02
MAX_POSITION_SIZE=0.1

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true

# Monitoring
ENABLE_DISCORD_NOTIFICATIONS=true
DISCORD_WEBHOOK_URL=your_webhook_url_here

# Performance
MAX_WORKERS=4
CACHE_SIZE=1000
EOF
        
        print_warn "Template configuration created. Please update with actual values."
        log "Template configuration created"
    fi
    
    save_state "config_setup" "completed"
}

# Function to deploy application
deploy_application() {
    print_step "Deploying application..."
    
    # Change to app directory
    cd "$APP_DIR"
    
    # If git repository, pull latest changes
    if [[ -d ".git" ]]; then
        print_info "Pulling latest changes from git repository..."
        sudo -u "$DEPLOYMENT_USER" git pull
    else
        print_warn "Not a git repository. Ensure application files are present in $APP_DIR"
    fi
    
    # Build Docker image
    if [[ -f "Dockerfile" ]]; then
        print_info "Building Docker image..."
        sudo -u "$DEPLOYMENT_USER" docker compose -f docker-compose.yml -f docker-compose.prod.yml build
        print_info "Docker image built successfully"
        log "Docker image built"
    else
        print_error "Dockerfile not found in $APP_DIR"
        return 1
    fi
    
    save_state "app_deploy" "completed"
}

# Function to start application
start_application() {
    print_step "Starting application..."
    
    cd "$APP_DIR"
    
    # Start containers
    sudo -u "$DEPLOYMENT_USER" docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    
    # Wait for containers to be healthy
    print_info "Waiting for containers to start..."
    sleep 10
    
    # Check container status
    local container_status=$(docker compose ps --format json | jq -r '.[0].State' 2>/dev/null || echo "unknown")
    
    if [[ "$container_status" == "running" ]]; then
        print_info "Application started successfully"
        log "Application started"
    else
        print_warn "Application may not be running correctly. Status: $container_status"
        print_warn "Check logs with: docker compose logs"
    fi
    
    save_state "app_start" "completed"
}

# Function to create backup
create_backup() {
    local backup_name="$1"
    print_step "Creating backup: $backup_name..."
    
    local backup_path="$BACKUP_DIR/$backup_name"
    mkdir -p "$backup_path"
    
    # Backup configuration
    if [[ -f "$APP_DIR/.env" ]]; then
        cp "$APP_DIR/.env" "$backup_path/"
    fi
    
    # Backup data directory
    if [[ -d "$APP_DIR/data" ]]; then
        cp -r "$APP_DIR/data" "$backup_path/"
    fi
    
    # Backup logs
    if [[ -d "$APP_DIR/logs" ]]; then
        cp -r "$APP_DIR/logs" "$backup_path/"
    fi
    
    # Create tarball
    tar -czf "$backup_path.tar.gz" -C "$BACKUP_DIR" "$backup_name"
    rm -rf "$backup_path"
    
    print_info "Backup created: $backup_path.tar.gz"
    log "Backup created: $backup_name"
}

# Function to rollback deployment
rollback() {
    print_error "Rolling back deployment..."
    log "Rollback initiated"
    
    # Stop application if running
    cd "$APP_DIR" 2>/dev/null || true
    docker compose down 2>/dev/null || true
    
    # Restore from latest backup
    local latest_backup=$(ls -t "$BACKUP_DIR"/*.tar.gz 2>/dev/null | head -1)
    
    if [[ -n "$latest_backup" ]]; then
        print_info "Restoring from backup: $latest_backup"
        local backup_name=$(basename "$latest_backup" .tar.gz)
        tar -xzf "$latest_backup" -C "$BACKUP_DIR"
        
        # Restore configuration
        if [[ -f "$BACKUP_DIR/$backup_name/.env" ]]; then
            cp "$BACKUP_DIR/$backup_name/.env" "$APP_DIR/"
        fi
        
        # Restore data
        if [[ -d "$BACKUP_DIR/$backup_name/data" ]]; then
            rm -rf "$APP_DIR/data"
            cp -r "$BACKUP_DIR/$backup_name/data" "$APP_DIR/"
        fi
        
        print_info "Rollback completed. Please verify system state."
        log "Rollback completed"
    else
        print_warn "No backup found for rollback"
        log "Rollback failed - no backup found"
    fi
}

# Function to verify deployment
verify_deployment() {
    print_step "Verifying deployment..."
    
    local verification_failed=0
    
    # Check Docker service
    if ! systemctl is-active --quiet docker; then
        print_error "Docker service is not running"
        verification_failed=1
    else
        print_info "✓ Docker service is running"
    fi
    
    # Check firewall
    if ! ufw status | grep -q "Status: active"; then
        print_warn "Firewall is not active"
    else
        print_info "✓ Firewall is active"
    fi
    
    # Check application directory
    if [[ ! -d "$APP_DIR" ]]; then
        print_error "Application directory not found"
        verification_failed=1
    else
        print_info "✓ Application directory exists"
    fi
    
    # Check deployment user
    if ! id "$DEPLOYMENT_USER" &>/dev/null; then
        print_error "Deployment user not found"
        verification_failed=1
    else
        print_info "✓ Deployment user exists"
    fi
    
    # Check Docker containers
    cd "$APP_DIR" 2>/dev/null || true
    local running_containers=$(docker compose ps --format json 2>/dev/null | jq -r 'length' 2>/dev/null || echo "0")
    if [[ "$running_containers" -gt 0 ]]; then
        print_info "✓ Docker containers are running ($running_containers)"
    else
        print_warn "No Docker containers running"
    fi
    
    if [[ $verification_failed -eq 0 ]]; then
        print_info "Deployment verification passed"
        log "Deployment verification passed"
        return 0
    else
        print_error "Deployment verification failed"
        log "Deployment verification failed"
        return 1
    fi
}

# Function to display deployment summary
display_summary() {
    echo ""
    echo "========================================"
    echo "  Deployment Summary"
    echo "========================================"
    echo "Application Directory: $APP_DIR"
    echo "Deployment User: $DEPLOYMENT_USER"
    echo "Log Directory: $LOG_DIR"
    echo "Backup Directory: $BACKUP_DIR"
    echo ""
    echo "Next Steps:"
    echo "1. Update configuration: $APP_DIR/.env"
    echo "2. Review logs: docker compose logs -f"
    echo "3. Monitor application: docker compose ps"
    echo "4. Access application: Check configured ports"
    echo ""
    echo "Useful Commands:"
    echo "  docker compose ps                  # Check container status"
    echo "  docker compose logs -f             # View logs"
    echo "  docker compose restart             # Restart application"
    echo "  docker compose down                # Stop application"
    echo ""
    echo "Deployment completed at: $(date)"
    echo "========================================"
}

# Main deployment function
main() {
    echo "========================================"
    echo "  Vultr VM Deployment Script"
    echo "  Version: $SCRIPT_VERSION"
    echo "========================================"
    echo ""
    
    # Create backup before deployment
    local backup_name="pre-deploy-$(date +%Y%m%d-%H%M%S)"
    if [[ -d "$APP_DIR" ]]; then
        create_backup "$backup_name"
    fi
    
    # Trap errors for rollback
    trap 'print_error "Deployment failed at step: $(cat $STATE_FILE 2>/dev/null | jq -r .step || echo unknown)"; rollback; exit 1' ERR
    
    # Run deployment steps
    check_root
    setup_logging
    check_requirements
    install_essential_tools
    install_docker
    create_deployment_user
    setup_ssh_keys
    configure_firewall
    create_app_directories
    setup_app_config
    
    # Optional: deploy and start application if files are present
    if [[ -d "$APP_DIR" ]] && [[ -f "$APP_DIR/docker-compose.yml" ]]; then
        deploy_application
        start_application
    else
        print_warn "Application files not found. Skipping deployment and start."
        print_warn "Please copy application files to $APP_DIR and run deployment manually."
    fi
    
    # Verify deployment
    verify_deployment
    
    # Display summary
    display_summary
    
    # Save final state
    save_state "deployment_complete" "success"
    log "Deployment completed successfully"
    
    print_info "Deployment completed successfully!"
}

# Parse command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    rollback)
        rollback
        ;;
    verify)
        verify_deployment
        ;;
    backup)
        backup_name="${2:-manual-$(date +%Y%m%d-%H%M%S)}"
        create_backup "$backup_name"
        ;;
    --help|-h)
        echo "Usage: $0 {deploy|rollback|verify|backup [name]}"
        echo ""
        echo "Commands:"
        echo "  deploy          Full deployment (default)"
        echo "  rollback        Rollback to previous state"
        echo "  verify          Verify deployment status"
        echo "  backup [name]   Create backup"
        echo ""
        echo "Environment Variables:"
        echo "  DEPLOYMENT_USER           User for deployment (default: tradingbot)"
        echo "  APP_DIR                   Application directory (default: /opt/trading-bot)"
        echo "  BACKUP_DIR                Backup directory (default: /opt/trading-bot-backups)"
        echo "  LOG_DIR                   Log directory (default: /var/log/trading-bot)"
        echo "  CONFIG_FILE               Config file to deploy (default: .env.production)"
        echo "  SSH_KEY_PATH              SSH public key path (default: ~/.ssh/id_rsa.pub)"
        echo "  APP_PORT                  Application port for firewall (optional)"
        exit 0
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 --help' for usage information"
        exit 1
        ;;
esac
