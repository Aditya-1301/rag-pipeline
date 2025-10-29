# 🚀 Docker Deployment Guide - RAG Terminal

Complete guide for deploying the RAG Terminal application using Docker and Docker Compose.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Building the Image](#building-the-image)
- [Running with Docker Compose](#running-with-docker-compose)
- [Environment Variables](#environment-variables)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Docker**: 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: 2.0+ ([Install Docker Compose](https://docs.docker.com/compose/install/))
- **Minimum Resources**: 4GB RAM, 10GB disk space
- **API Keys** (optional but recommended for better performance):
  - HuggingFace Token (for embeddings)
  - Ollama Cloud API Key (for cloud LLM)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Aditya-1301/rag-pipeline.git
cd rag-pipeline
```

### 2. Create Environment File

```bash
cp .env.example .env
nano .env  # Edit with your API keys
```

### 3. Start the Application

```bash
# Start both Ollama and RAG app
docker-compose up -d

# View logs
docker-compose logs -f

# Access the application
# Open browser: http://localhost:7860
```

### 4. Initial Setup (First Run)

```bash
# Pull Ollama models
docker exec -it rag-ollama ollama pull smollm2:360m
docker exec -it rag-ollama ollama pull gpt-oss:20b-cloud

# Check status
docker-compose ps
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Ollama Configuration
OLLAMA_MODEL=smollm2:360m
OLLAMA_MODEL_CLOUD=gpt-oss:20b-cloud
OLLAMA_API_KEY=your_ollama_cloud_api_key_here

# Embedding Configuration
EMBEDDING_METHOD=huggingface  # Options: huggingface, voyage, openai, fastembed
HF_TOKEN=hf_your_token_here
OPENAI_API_KEY=sk-your-key-here
VOYAGE_API_KEY=pa-your-key-here
HF_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Gradio Configuration (optional)
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

### Document Management

Place your documents in the mounted directory:

```bash
# Default mount: ~/Documents/Books
# Or customize in docker-compose.yml:
volumes:
  - /path/to/your/documents:/app/documents:ro
```

## 🔨 Building the Image

### Build Locally

```bash
# Build the image
docker build -t rag-terminal:latest .

# Build with custom tags
docker build -t rag-terminal:v1.0.0 .

# Build without cache
docker build --no-cache -t rag-terminal:latest .
```

### Multi-Platform Build

```bash
# Set up buildx
docker buildx create --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t rag-terminal:latest \
  --push .
```

## 🐳 Running with Docker Compose

### Start Services

```bash
# Start in detached mode
docker-compose up -d

# Start with rebuild
docker-compose up -d --build

# Start specific service
docker-compose up -d ollama
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v

# Stop and remove images
docker-compose down --rmi all
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rag-app

# Last 100 lines
docker-compose logs --tail=100 rag-app
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart rag-app
```

## 📊 Monitoring

### Health Checks

```bash
# Check container status
docker-compose ps

# Check health status
docker inspect --format='{{.State.Health.Status}}' rag-terminal

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' rag-terminal
```

### Resource Usage

```bash
# Monitor resource usage
docker stats rag-terminal rag-ollama

# Disk usage
docker system df
```

## 🌐 Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml rag-stack

# List services
docker stack services rag-stack

# Remove stack
docker stack rm rag-stack
```

### Using Kubernetes

```bash
# Generate Kubernetes manifests
kompose convert -f docker-compose.yml

# Apply to cluster
kubectl apply -f .

# Check deployment
kubectl get pods
kubectl get services
```

### Behind Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/rag-terminal
server {
    listen 80;
    server_name rag.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL with Certbot

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d rag.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Find process using port 7860
sudo lsof -i :7860

# Kill process
sudo kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8080:7860"  # Use port 8080 instead
```

#### 2. Out of Disk Space

```bash
# Clean up Docker
docker system prune -a --volumes

# Remove old images
docker image prune -a

# Remove unused volumes
docker volume prune
```

#### 3. Ollama Model Not Found

```bash
# Pull model manually
docker exec -it rag-ollama ollama pull smollm2:360m

# List available models
docker exec -it rag-ollama ollama list
```

#### 4. Permission Issues

```bash
# Fix permissions for mounted directories
sudo chown -R $(id -u):$(id -g) ./rag_data
sudo chown -R $(id -u):$(id -g) ./.embedding_cache

# Run with user ID
docker-compose run --user $(id -u):$(id -g) rag-app
```

#### 5. Memory Issues

```bash
# Limit memory in docker-compose.yml
services:
  rag-app:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### Debug Mode

```bash
# Run with debug output
docker-compose up

# Enter container for debugging
docker exec -it rag-terminal bash

# Check Python environment
docker exec -it rag-terminal python --version
docker exec -it rag-terminal pip list
```

### Logs Location

```bash
# View container logs
docker logs rag-terminal

# Export logs
docker logs rag-terminal > app.log 2>&1

# Follow logs in real-time
docker logs -f rag-terminal
```

## 📈 Performance Optimization

### 1. Use Pre-computed Embeddings

Mount existing embeddings to avoid recomputation:

```yaml
volumes:
  - ./rag_data:/app/rag_data
  - ./.embedding_cache:/app/.embedding_cache
```

### 2. Enable GPU Support (NVIDIA)

```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. Optimize Image Size

- Use multi-stage builds (already implemented)
- Use CPU-only PyTorch
- Remove unnecessary dependencies

### 4. Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

## 🔐 Security Best Practices

1. **Never commit `.env` files**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use secrets for production**
   ```bash
   docker secret create ollama_api_key ./api_key.txt
   ```

3. **Run as non-root user**
   ```dockerfile
   RUN useradd -m -u 1000 rag
   USER rag
   ```

4. **Keep images updated**
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

5. **Scan for vulnerabilities**
   ```bash
   docker scan rag-terminal:latest
   ```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/docker.md)
- [Gradio Deployment Guide](https://gradio.app/guides/deploying-gradio-with-docker/)

## 🆘 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs: `docker-compose logs -f`
3. Open an issue on GitHub
4. Join the community Discord

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
