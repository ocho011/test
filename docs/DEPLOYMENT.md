# Vultr VM Deployment Guide

Complete guide for deploying the ICT Trading Bot to Vultr VMs using the automated deployment script.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment Process](#deployment-process)
- [Post-Deployment](#post-deployment)
- [Maintenance](#maintenance)
- [Troubleshooting](#troubleshooting)
- [Security Best Practices](#security-best-practices)

## Overview

The deployment script (`scripts/deploy.sh`) automates the complete setup of the ICT Trading Bot on Vultr VMs, including:

- ✅ Docker and Docker Compose installation
- ✅ System dependencies and essential tools
- ✅ SSH key-based authentication
- ✅ Firewall configuration (UFW)
- ✅ Application directory structure
- ✅ Automated backup before deployment
- ✅ Rollback capability on failure
- ✅ Comprehensive deployment verification

## Prerequisites

### On Your Local Machine

1. **SSH Key Pair**: Generate if you don't have one
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```

2. **Project Files**: Ensure you have the complete project directory ready
   ```bash
   git clone <your-repo-url>
   cd trading-bot
   ```

3. **Configuration Files**: Prepare your production configuration
   ```bash
   cp .env.production.example .env.production
   # Edit .env.production with your API keys and settings
   ```

### On Vultr

1. **VM Instance**: Create a Vultr VM with:
   - OS: Ubuntu 22.04 LTS (recommended) or Ubuntu 20.04 LTS
   - Memory: 2GB minimum (4GB recommended)
   - Storage: 50GB minimum
   - Region: Choose closest to your trading exchange

2. **Initial Access**: Note your VM's IP address and root password

## Quick Start

### Step 1: Initial VM Setup

SSH into your Vultr VM:
```bash
ssh root@your-vm-ip
```

### Step 2: Upload Deployment Script

From your local machine:
```bash
# Upload the deployment script
scp scripts/deploy.sh root@your-vm-ip:/tmp/

# Upload configuration files
scp .env.production root@your-vm-ip:/tmp/
scp config/deploy.config.example root@your-vm-ip:/tmp/deploy.config
```

### Step 3: Configure Deployment

On the VM, edit the deployment configuration:
```bash
nano /tmp/deploy.config
```

Set your SSH public key path:
```bash
export SSH_KEY_PATH="/root/.ssh/authorized_keys"
```

### Step 4: Run Deployment

Execute the deployment script:
```bash
# Source configuration
source /tmp/deploy.config

# Run deployment
chmod +x /tmp/deploy.sh
/tmp/deploy.sh deploy
```

### Step 5: Upload Application Files

After infrastructure is ready, upload your application:
```bash
# From local machine
scp -r . tradingbot@your-vm-ip:/opt/trading-bot/
```

Or clone from git:
```bash
# On the VM as tradingbot user
sudo -u tradingbot -i
cd /opt/trading-bot
git clone <your-repo-url> .
```

### Step 6: Start Application

```bash
cd /opt/trading-bot
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Configuration

### Environment Variables

The deployment script can be customized using environment variables:

```bash
# Deployment User
export DEPLOYMENT_USER="tradingbot"           # User for running application

# Directories
export APP_DIR="/opt/trading-bot"             # Application directory
export BACKUP_DIR="/opt/trading-bot-backups"  # Backup storage
export LOG_DIR="/var/log/trading-bot"         # Log directory

# Configuration
export CONFIG_FILE=".env.production"          # Production config file
export SSH_KEY_PATH="$HOME/.ssh/id_rsa.pub"   # SSH public key

# System
export DOCKER_VERSION="latest"                # Docker version
export REQUIRED_DISK_SPACE_GB="10"            # Minimum disk space

# Application (optional)
export APP_PORT="8000"                        # Application port for firewall
```

### Configuration File

Create `/tmp/deploy.config` with your settings:

```bash
cp config/deploy.config.example /tmp/deploy.config
nano /tmp/deploy.config
```

## Deployment Process

### What the Script Does

The deployment script executes these steps in order:

1. **Pre-Deployment Backup**
   - Creates backup of existing installation (if any)
   - Stores backup in timestamped archive

2. **System Requirements Check**
   - Verifies OS compatibility
   - Checks available disk space
   - Validates memory availability

3. **Docker Installation**
   - Installs Docker Engine and Docker Compose
   - Configures Docker daemon
   - Enables Docker service

4. **User Setup**
   - Creates dedicated deployment user
   - Adds user to docker group
   - Sets up proper permissions

5. **SSH Security**
   - Configures SSH key-based authentication
   - Disables password authentication
   - Restarts SSH daemon

6. **Firewall Configuration**
   - Installs and configures UFW
   - Opens required ports (SSH, HTTP, HTTPS)
   - Enables firewall with safe defaults

7. **Essential Tools**
   - Installs git, vim, curl, wget, jq
   - Installs Python and build tools
   - Installs monitoring utilities

8. **Directory Structure**
   - Creates application directories
   - Sets proper ownership and permissions
   - Creates log and data directories

9. **Application Configuration**
   - Deploys production configuration
   - Sets secure file permissions
   - Creates template if needed

10. **Verification**
    - Checks all services are running
    - Verifies configurations
    - Tests connectivity

### Manual Deployment Steps

If you prefer manual deployment or need to customize:

```bash
# 1. Source configuration
source /tmp/deploy.config

# 2. Run individual steps
/tmp/deploy.sh deploy          # Full deployment
# OR run components separately:
# install_docker
# create_deployment_user
# setup_ssh_keys
# configure_firewall
# etc.
```

## Post-Deployment

### Verify Installation

Check deployment status:
```bash
/tmp/deploy.sh verify
```

Expected output:
```
✓ Docker service is running
✓ Firewall is active
✓ Application directory exists
✓ Deployment user exists
✓ Docker containers are running
```

### Access Your Application

1. **SSH Access** (as deployment user):
   ```bash
   ssh tradingbot@your-vm-ip
   ```

2. **Check Docker Containers**:
   ```bash
   docker compose ps
   ```

3. **View Logs**:
   ```bash
   docker compose logs -f
   ```

4. **Monitor Application**:
   ```bash
   # System resources
   htop
   
   # Docker stats
   docker stats
   
   # Application logs
   tail -f /opt/trading-bot/logs/trading.log
   ```

### Configure Monitoring

Set up Discord notifications (optional):
```bash
# Edit .env file
nano /opt/trading-bot/.env

# Update Discord webhook
ENABLE_DISCORD_NOTIFICATIONS=true
DISCORD_WEBHOOK_URL=your_webhook_url_here

# Restart application
docker compose restart
```

## Maintenance

### Regular Backups

Create manual backup:
```bash
sudo /tmp/deploy.sh backup
```

Automated backups are created before each deployment.

### Updates and Redeployment

Update application:
```bash
# As tradingbot user
cd /opt/trading-bot
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml build
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Rollback

If deployment fails or issues occur:
```bash
sudo /tmp/deploy.sh rollback
```

This restores the previous backup.

### Log Management

View deployment logs:
```bash
cat /var/log/trading-bot/deployment.log
```

View application logs:
```bash
docker compose logs -f
# OR
tail -f /opt/trading-bot/logs/trading.log
```

### System Updates

Keep the system updated:
```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Update Docker
sudo apt install docker-ce docker-ce-cli containerd.io
```

## Troubleshooting

### Common Issues

#### 1. SSH Connection Refused

**Problem**: Cannot connect via SSH after deployment

**Solution**:
```bash
# On VM console (via Vultr web console)
systemctl status sshd
systemctl restart sshd

# Check SSH configuration
cat /etc/ssh/sshd_config | grep -E "PasswordAuthentication|PubkeyAuthentication"
```

#### 2. Firewall Blocks Access

**Problem**: Cannot access application after firewall setup

**Solution**:
```bash
# Check firewall status
sudo ufw status

# Open required port
sudo ufw allow 8000/tcp

# Reload firewall
sudo ufw reload
```

#### 3. Docker Service Not Running

**Problem**: Docker containers won't start

**Solution**:
```bash
# Check Docker status
systemctl status docker

# Start Docker
systemctl start docker

# Check Docker logs
journalctl -u docker -n 50
```

#### 4. Insufficient Permissions

**Problem**: Permission denied errors

**Solution**:
```bash
# Fix application directory ownership
sudo chown -R tradingbot:tradingbot /opt/trading-bot

# Add user to docker group (re-login required)
sudo usermod -aG docker tradingbot
```

#### 5. Out of Disk Space

**Problem**: Deployment fails due to disk space

**Solution**:
```bash
# Check disk usage
df -h

# Clean Docker resources
docker system prune -a --volumes

# Remove old logs
sudo find /var/log -type f -name "*.log" -mtime +30 -delete
```

### Debugging Steps

1. **Check Deployment Log**:
   ```bash
   cat /var/log/trading-bot/deployment.log
   ```

2. **Check Deployment State**:
   ```bash
   cat /tmp/deployment_state.json
   ```

3. **Verify System Requirements**:
   ```bash
   # Check OS
   cat /etc/os-release
   
   # Check disk space
   df -h
   
   # Check memory
   free -h
   
   # Check Docker
   docker --version
   docker compose version
   ```

4. **Test Network Connectivity**:
   ```bash
   # Test Binance API
   curl -I https://api.binance.com/api/v3/ping
   
   # Test DNS
   nslookup api.binance.com
   ```

## Security Best Practices

### 1. SSH Security

- ✅ Use SSH keys only (password auth disabled)
- ✅ Change default SSH port (optional):
  ```bash
  # Edit SSH config
  sudo nano /etc/ssh/sshd_config
  # Change: Port 22 → Port 2222
  sudo systemctl restart sshd
  
  # Update firewall
  sudo ufw allow 2222/tcp
  sudo ufw delete allow 22/tcp
  ```

### 2. API Key Security

- ✅ Never commit API keys to git
- ✅ Use environment-specific .env files
- ✅ Set restrictive file permissions:
  ```bash
  chmod 600 /opt/trading-bot/.env
  ```

### 3. Firewall Rules

- ✅ Only open required ports
- ✅ Use IP whitelisting for sensitive ports
- ✅ Regular audit of firewall rules:
  ```bash
  sudo ufw status numbered
  ```

### 4. Docker Security

- ✅ Run containers as non-root user
- ✅ Limit container resources
- ✅ Regular security updates:
  ```bash
  docker compose pull
  docker compose up -d
  ```

### 5. Monitoring

- ✅ Enable application logging
- ✅ Set up Discord/Slack notifications
- ✅ Monitor system resources
- ✅ Regular backup verification

### 6. Regular Updates

- ✅ Keep OS updated
- ✅ Update Docker regularly
- ✅ Update application dependencies
- ✅ Review security advisories

## Additional Resources

- [Vultr Documentation](https://www.vultr.com/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [UFW Guide](https://help.ubuntu.com/community/UFW)
- [SSH Security Best Practices](https://www.ssh.com/academy/ssh/security)

## Support

If you encounter issues:

1. Check this documentation
2. Review deployment logs
3. Verify system requirements
4. Check the troubleshooting section
5. Create an issue in the project repository

---

**Last Updated**: 2025
**Script Version**: 1.0.0
