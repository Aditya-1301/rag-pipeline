# 🚀 RAG Terminal Deployment Guide

## Quick Start

### Local Testing (Requires Docker)

```bash
# 1. Clone the repository
git clone https://github.com/Aditya-1301/rag-pipeline.git
cd rag-pipeline

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys (optional - app works with local embeddings)
nano .env

# 3. Build Docker image
docker compose build

# 4. Start services
docker compose up

# 5. Access the app
# Gradio UI: http://localhost:7860
# Ollama: http://localhost:11434
```

## Deployment Options

### Option 1: Local Machine (Development)

**Requirements:**
- Python 3.12+
- Ollama (for local LLM)
- HuggingFace API token (optional, for faster embeddings)

**Steps:**

```bash
# 1. Clone repo
git clone https://github.com/Aditya-1301/rag-pipeline.git
cd rag-pipeline

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Set up environment
cp .env.example .env
nano .env  # Add your API keys

# 5. Ensure Ollama is running
ollama serve  # In another terminal

# 6. Run the app
python app.py

# 7. Open http://localhost:7860
```

### Option 2: Docker Compose (Recommended for Production)

**Requirements:**
- Docker & Docker Compose
- 4GB+ RAM
- 2GB disk space for models

**Steps:**

```bash
# 1. Clone repo
git clone https://github.com/Aditya-1301/rag-pipeline.git
cd rag-pipeline

# 2. Set up environment
cp .env.example .env
# Edit .env to add API keys (optional)
nano .env

# 3. Build and start
docker compose build
docker compose up -d

# 4. Check logs
docker compose logs -f rag-app
docker compose logs -f ollama

# 5. Access http://localhost:7860

# 6. Stop services
docker compose down
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

#### AWS EC2 Deployment

```bash
# 1. Launch EC2 instance (Ubuntu 22.04, t3.medium or larger)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance.com

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 4. Clone and deploy
git clone https://github.com/Aditya-1301/rag-pipeline.git
cd rag-pipeline
sudo nano .env  # Add API keys
sudo docker compose up -d

# 5. Configure firewall (AWS Security Group)
# Open port 7860 for Gradio UI
# Open port 11434 for Ollama (internal only)

# 6. Set up reverse proxy (optional, for domain)
sudo apt install nginx
# Configure nginx to reverse proxy to localhost:7860
```

#### Using GitHub Container Registry (ghcr.io)

```bash
# 1. Enable GitHub Container Registry
# Settings → Developer settings → Personal access tokens → Generate token
# Select: read:packages, write:packages, delete:packages

# 2. Push to registry
docker tag rag-terminal ghcr.io/your-username/rag-terminal:latest
docker push ghcr.io/your-username/rag-terminal:latest

# 3. Pull on deployment server
docker pull ghcr.io/your-username/rag-terminal:latest
docker run -p 7860:7860 ghcr.io/your-username/rag-terminal:latest
```

### Option 4: Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-terminal
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rag-terminal
  template:
    metadata:
      labels:
        app: rag-terminal
    spec:
      containers:
      - name: rag-app
        image: ghcr.io/your-username/rag-terminal:latest
        ports:
        - containerPort: 7860
        env:
        - name: OLLAMA_BASE_URL
          value: "http://ollama:11434"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: hf-token
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: rag-data
          mountPath: /app/rag_data
      volumes:
      - name: rag-data
        persistentVolumeClaim:
          claimName: rag-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: rag-terminal
spec:
  type: LoadBalancer
  ports:
  - port: 7860
    targetPort: 7860
  selector:
    app: rag-terminal
```

Deploy with:
```bash
kubectl apply -f deployment.yaml
kubectl get svc rag-terminal  # Get external IP
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_METHOD` | `huggingface` | Embedding backend: huggingface, voyage, openai, fastembed |
| `HF_TOKEN` | - | HuggingFace API token (get from huggingface.co) |
| `OPENAI_API_KEY` | - | OpenAI API key (for OpenAI embeddings) |
| `VOYAGE_API_KEY` | - | Voyage AI API key (for Voyage embeddings) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `smollm2:360m` | Local Ollama model |
| `OLLAMA_MODEL_CLOUD` | `gpt-oss:20b-cloud` | Cloud Ollama model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Number of sources to retrieve |
| `GRADIO_SERVER_PORT` | `7860` | Gradio UI port |

## Performance Optimization

### CPU-Only Mode (Default)
- Uses CPU-only PyTorch (~184MB)
- Suitable for most deployments
- No GPU required

### GPU Support (Optional)

For GPU acceleration, modify `docker-compose.yml`:

```yaml
services:
  rag-app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

### Caching Strategy

The app caches embeddings to speed up subsequent queries:
- **First query on document:** ~5 minutes (generates embeddings)
- **Subsequent queries:** <5 seconds (cached)
- Cache location: `./rag_data/` (mounted volume in Docker)

## Monitoring & Logs

### Docker Compose
```bash
# View live logs
docker compose logs -f rag-app

# View specific service
docker compose logs ollama

# Last 100 lines
docker compose logs --tail 100
```

### Health Checks
```bash
# Check Gradio
curl http://localhost:7860/

# Check Ollama
curl http://localhost:11434/api/tags

# Docker health status
docker compose ps
```

## Troubleshooting

### "Port already in use"
```bash
# Find process using port 7860
lsof -i :7860

# Kill it
kill -9 <PID>

# Or use different port
docker compose -p rag-terminal-2 up
```

### "Out of memory"
- Reduce chunk size: `CHUNK_SIZE=500`
- Use smaller model: `OLLAMA_MODEL=qwen2.5:0.5b`
- Limit top_k: `TOP_K=3`

### "Slow embedding generation"
- Use API-based embeddings (HuggingFace, OpenAI)
- `EMBEDDING_METHOD=huggingface`
- Add API token: `HF_TOKEN=...`

### "Ollama connection refused"
```bash
# Make sure Ollama is running
docker compose ps ollama

# Check Ollama logs
docker compose logs ollama

# Restart Ollama
docker compose restart ollama
```

## Production Checklist

- [ ] Set `GRADIO_SHARE=false` (secure mode)
- [ ] Use HTTPS/SSL certificates (nginx/Let's Encrypt)
- [ ] Set resource limits (memory, CPU, disk)
- [ ] Configure health checks and auto-restart
- [ ] Set up monitoring and alerting
- [ ] Regular backups of `./rag_data/`
- [ ] API rate limiting (nginx/reverse proxy)
- [ ] Container security scanning (Trivy)
- [ ] Environment-specific configs (.env per environment)
- [ ] Log aggregation (optional: ELK stack, CloudWatch)

## Scaling

### Horizontal Scaling (Multiple Instances)

Use a load balancer (nginx, HAProxy) to distribute traffic:

```nginx
upstream rag_backend {
    server localhost:7860;
    server localhost:7861;
    server localhost:7862;
}

server {
    listen 80;
    server_name rag.yourdomain.com;
    
    location / {
        proxy_pass http://rag_backend;
    }
}
```

Each instance shares the same `rag_data` volume for embeddings.

### Vertical Scaling (Larger Instance)

- Increase memory: `2GB → 8GB+`
- Add more CPU cores: `2 → 8 cores`
- Use GPU: NVIDIA GPU for faster inference

## CI/CD Integration

GitHub Actions automatically:
1. ✅ Tests code (pytest)
2. ✅ Builds Docker image
3. ✅ Scans security (Trivy)
4. ✅ Pushes to ghcr.io
5. ✅ Deploys to production

Workflow: `.github/workflows/ci-cd.yml`

## Support & Documentation

- **README**: `/README.md` - Project overview
- **Multi-Document Update**: `/MULTI_DOCUMENT_UPDATE.md` - UI changes
- **Requirements**: `/requirements.txt` - Dependencies
- **Docker**: `/Dockerfile`, `/docker-compose.yml` - Container setup
- **GitHub Actions**: `/.github/workflows/ci-cd.yml` - CI/CD config

## Quick Reference

```bash
# Development
python app.py

# Docker local
docker compose up

# Docker production
docker compose -f docker-compose.prod.yml up -d

# Stop all
docker compose down

# Rebuild
docker compose build --no-cache

# View config
docker compose config

# Execute in container
docker compose exec rag-app bash

# Access Ollama
docker compose exec ollama ollama list

# View volumes
docker volume ls
docker volume inspect rag-terminal_rag_data
```

---

**Questions?** Open an issue on GitHub or check the troubleshooting section above.
