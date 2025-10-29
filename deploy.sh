#!/bin/bash

# RAG Terminal - Docker Deployment Script
# This script simplifies common Docker operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if .env file exists
check_env() {
    if [ ! -f .env ]; then
        print_info "No .env file found. Creating from .env.example..."
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file. Please edit it with your API keys."
            print_info "Run: nano .env"
            exit 0
        else
            print_error "No .env.example file found."
            exit 1
        fi
    fi
    print_success ".env file exists"
}

# Build the Docker image
build() {
    print_info "Building Docker image..."
    docker-compose build
    print_success "Build completed"
}

# Start the services
start() {
    print_info "Starting services..."
    docker-compose up -d
    print_success "Services started"
    
    print_info "Waiting for services to be healthy..."
    sleep 5
    
    docker-compose ps
    
    print_info "\nAccess the application at: http://localhost:7860"
    print_info "View logs with: ./deploy.sh logs"
}

# Stop the services
stop() {
    print_info "Stopping services..."
    docker-compose down
    print_success "Services stopped"
}

# Restart the services
restart() {
    print_info "Restarting services..."
    docker-compose restart
    print_success "Services restarted"
}

# View logs
logs() {
    docker-compose logs -f "${1:-rag-app}"
}

# Pull Ollama models
pull_models() {
    print_info "Pulling Ollama models..."
    
    # Check if Ollama container is running
    if ! docker ps | grep -q rag-ollama; then
        print_error "Ollama container is not running. Start services first."
        exit 1
    fi
    
    # Pull default models
    print_info "Pulling smollm2:360m..."
    docker exec -it rag-ollama ollama pull smollm2:360m
    
    print_info "Pulling gpt-oss:20b-cloud..."
    docker exec -it rag-ollama ollama pull gpt-oss:20b-cloud || print_info "Cloud model pull failed (may require authentication)"
    
    print_success "Models pulled"
    
    # List available models
    print_info "Available models:"
    docker exec -it rag-ollama ollama list
}

# Check status
status() {
    print_info "Service status:"
    docker-compose ps
    
    print_info "\nResource usage:"
    docker stats --no-stream rag-terminal rag-ollama 2>/dev/null || print_info "Containers not running"
}

# Clean up
clean() {
    print_info "Cleaning up Docker resources..."
    docker-compose down -v
    docker system prune -f
    print_success "Cleanup completed"
}

# Full setup (for first-time deployment)
setup() {
    print_info "Starting full setup..."
    
    check_docker
    check_env
    
    print_info "Building images..."
    build
    
    print_info "Starting services..."
    start
    
    print_info "Pulling Ollama models..."
    sleep 10  # Wait for Ollama to be ready
    pull_models
    
    print_success "Setup completed!"
    print_info "\n🚀 Your RAG Terminal is ready!"
    print_info "   Access at: http://localhost:7860"
    print_info "   View logs: ./deploy.sh logs"
}

# Show help
help() {
    cat << EOF
RAG Terminal - Docker Deployment Script

Usage: ./deploy.sh [command]

Commands:
    setup       - Full setup (first-time deployment)
    build       - Build Docker images
    start       - Start all services
    stop        - Stop all services
    restart     - Restart all services
    logs [svc]  - View logs (optional: specify service)
    models      - Pull Ollama models
    status      - Check service status
    clean       - Clean up Docker resources
    help        - Show this help message

Examples:
    ./deploy.sh setup          # First-time setup
    ./deploy.sh start          # Start services
    ./deploy.sh logs           # View app logs
    ./deploy.sh logs ollama    # View Ollama logs
    ./deploy.sh status         # Check status

Environment:
    Edit .env file to configure API keys and settings
    
EOF
}

# Main script
case "${1:-help}" in
    setup)
        setup
        ;;
    build)
        check_docker
        build
        ;;
    start)
        check_docker
        check_env
        start
        ;;
    stop)
        check_docker
        stop
        ;;
    restart)
        check_docker
        restart
        ;;
    logs)
        check_docker
        logs "$2"
        ;;
    models)
        check_docker
        pull_models
        ;;
    status)
        check_docker
        status
        ;;
    clean)
        check_docker
        clean
        ;;
    help|*)
        help
        ;;
esac
