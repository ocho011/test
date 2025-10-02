# Docker Deployment Guide

This guide covers containerization and deployment of the ICT Trading Bot using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10 or later
- Docker Compose 2.0 or later
- At least 2GB free disk space
- Valid API keys configured in `.env` file

## Quick Start

### 1. Environment Setup

Copy the environment template and configure your API keys:

```bash
cp .env.template .env
# Edit .env with your actual API keys and configuration
```

### 2. Build and Run (Development)

```bash
# Build the Docker image
./scripts/docker_build.sh build dev

# Start the containers
./scripts/docker_build.sh start dev

# View logs
./scripts/docker_build.sh logs
```

### 3. Build and Run (Production)

```bash
# Build production image
./scripts/docker_build.sh build prod

# Start production containers
./scripts/docker_build.sh start prod

# View logs
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f
```

## Docker Images

### Multi-Stage Build Architecture

The Dockerfile uses a multi-stage build to optimize image size:

1. **Builder Stage**: Installs build dependencies and compiles packages
2. **Runtime Stage**: Minimal image with only runtime dependencies

This approach reduces the final image size by ~60% compared to a single-stage build.

### Image Specifications

- **Base Image**: `python:3.11-slim`
- **User**: Non-root user `tradingbot` (UID 1000)
- **Working Directory**: `/app`
- **Python Path**: `/app/src`

## Docker Compose Configuration

### Base Configuration (`docker-compose.yml`)

- Service orchestration for trading bot
- Volume mounts for logs, data, and configuration
- Network configuration
- Resource limits (2 CPU cores, 1GB RAM)
- Health checks every 30 seconds

### Development Override (`docker-compose.dev.yml`)

- Source code mounting for live reload
- Debugger port (5678) exposed
- Higher resource limits (4 CPU cores, 2GB RAM)
- Debug environment variables

### Production Override (`docker-compose.prod.yml`)

- Read-only configuration mounts
- Stricter resource limits
- Enhanced health checks (every 15 seconds)
- Log rotation and compression
- Automatic restart policies

## Usage Examples

### Development Workflow

```bash
# Start development environment with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run tests in container
./scripts/docker_build.sh test

# Access container shell
docker-compose exec trading-bot /bin/bash

# View real-time logs
docker-compose logs -f trading-bot
```

### Production Deployment

```bash
# Build optimized production image
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Start in detached mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check container health
docker-compose ps

# Monitor logs
docker-compose logs -f trading-bot
```

### Maintenance Commands

```bash
# Stop containers
./scripts/docker_build.sh stop

# Restart containers
./scripts/docker_build.sh restart prod

# Clean up all resources
./scripts/docker_build.sh clean

# Rebuild from scratch
docker-compose build --no-cache
```

## Volume Mounts

### Development Volumes

- `./src:/app/src` - Source code (live reload)
- `./tests:/app/tests` - Test files
- `./logs:/app/logs` - Application logs
- `./data:/app/data` - Market data cache
- `./config:/app/config` - Runtime configuration

### Production Volumes

- `./logs:/app/logs` - Application logs (read-write)
- `./data:/app/data` - Market data cache (read-write)
- `./config:/app/config:ro` - Configuration (read-only)

## Networking

### Default Network

- **Name**: `trading-network`
- **Driver**: Bridge
- **Services**: trading-bot, (optional: prometheus, grafana)

### Port Mappings

- `8000` - Health check and API endpoints
- `5678` - Debugger port (development only)

## Health Checks

The container includes built-in health checks:

- **Endpoint**: `http://localhost:8000/health`
- **Interval**: 30s (dev) / 15s (prod)
- **Timeout**: 10s (dev) / 5s (prod)
- **Retries**: 3 (dev) / 5 (prod)
- **Start Period**: 40s (dev) / 60s (prod)

Check health status:

```bash
docker-compose ps
docker inspect --format='{{.State.Health.Status}}' ict-trading-bot
```

## Resource Management

### Development Limits

- **CPU**: 4.0 cores max
- **Memory**: 2GB max

### Production Limits

- **CPU**: 2.0 cores limit, 1.0 reserved
- **Memory**: 1GB limit, 768MB reserved

Monitor resource usage:

```bash
docker stats ict-trading-bot
```

## Logging

### Log Configuration

- **Driver**: json-file
- **Max Size**: 10MB (dev) / 50MB (prod)
- **Max Files**: 3 (dev) / 5 (prod)
- **Compression**: Enabled in production

### Accessing Logs

```bash
# View all logs
docker-compose logs trading-bot

# Follow logs in real-time
docker-compose logs -f trading-bot

# Last 100 lines
docker-compose logs --tail=100 trading-bot

# Logs since timestamp
docker-compose logs --since 2024-01-01T00:00:00 trading-bot
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs for errors
docker-compose logs trading-bot

# Verify configuration
docker-compose config

# Check resource availability
docker system df
```

### Health Check Failing

```bash
# Check health endpoint directly
docker-compose exec trading-bot curl http://localhost:8000/health

# Inspect health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' ict-trading-bot
```

### Performance Issues

```bash
# Check resource usage
docker stats ict-trading-bot

# View system events
docker events --filter container=ict-trading-bot

# Analyze image layers
docker history ict-trading-bot
```

### Build Issues

```bash
# Clean build cache
docker builder prune

# Build with verbose output
docker-compose build --progress=plain --no-cache

# Check build context size
docker-compose build 2>&1 | grep "Sending build context"
```

## Security Considerations

1. **Non-root User**: Container runs as `tradingbot` (UID 1000)
2. **Read-only Mounts**: Production config mounted as read-only
3. **No Privileged Mode**: Containers run without elevated privileges
4. **Minimal Base Image**: Uses slim Python image to reduce attack surface
5. **Environment Variables**: Sensitive data passed via env_file, not hardcoded

## Optimization Tips

1. **Multi-stage Build**: Already implemented for smaller images
2. **Layer Caching**: Order Dockerfile commands from least to most frequently changed
3. **Build Context**: Use `.dockerignore` to exclude unnecessary files
4. **Health Checks**: Tune intervals based on application startup time
5. **Resource Limits**: Adjust based on actual usage patterns

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Build Docker image
  run: |
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml build
    
- name: Run tests
  run: |
    docker-compose run --rm trading-bot python -m pytest
    
- name: Push to registry
  run: |
    docker tag ict-trading-bot:latest registry.example.com/ict-trading-bot:${{ github.sha }}
    docker push registry.example.com/ict-trading-bot:${{ github.sha }}
```

## Monitoring (Optional)

Uncomment Prometheus and Grafana services in `docker-compose.yml`:

```bash
# Start with monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana at http://localhost:3000
# Access Prometheus at http://localhost:9090
```

## Next Steps

- [VM Deployment Guide](./deployment.md) - Deploy to Vultr VM
- [Operations Manual](./operations.md) - Day-to-day operations
- [Development Guide](./development.md) - Local development setup
