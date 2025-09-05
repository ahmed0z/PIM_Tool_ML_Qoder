# AutoPatternChecker Deployment Guide

This guide covers deployment options for AutoPatternChecker in different environments.

## Development Deployment (Google Colab)

### Prerequisites
- Google Colab account
- CSV data file
- Google Drive (optional, for saving artifacts)

### Steps
1. Open `colab_notebook_full_pipeline.ipynb` in Google Colab
2. Upload your CSV file when prompted
3. Run all cells sequentially
4. The API will be available at `http://localhost:8000`
5. Use ngrok or Colab's port forwarding to expose the API externally

### Saving Artifacts
The notebook automatically saves all generated artifacts to Google Drive:
- Key profiles (`key_profiles.json`)
- Normalization rules (`normalization_rules.json`)
- Clustering results (`clustering_results.json`)
- Embeddings data (`embeddings_data.pkl`)
- FAISS indices (`indices/` directory)

## Local Development

### Prerequisites
- Python 3.10+
- 8GB+ RAM recommended
- 10GB+ free disk space

### Installation
```bash
# Clone repository
git clone <repository-url>
cd autopatternchecker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest
```

### Running the Service
```bash
# Start the API server
uvicorn autopatternchecker.api:app --host 0.0.0.0 --port 8000 --reload

# Or use the provided script
python -m autopatternchecker.api
```

## Docker Deployment

### Single Container

```bash
# Build the image
docker build -t autopatternchecker:latest .

# Run the container
docker run -d \
  --name autopatternchecker \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/indices:/app/indices \
  autopatternchecker:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Compose with Nginx

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  autopatternchecker:
    build: .
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./indices:/app/indices
      - ./logs:/app/logs
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - autopatternchecker
    restart: unless-stopped
```

## Production Deployment

### Server Requirements

#### Minimum Requirements
- **CPU**: 4 vCPUs
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **OS**: Ubuntu 20.04+ or CentOS 8+

#### Recommended for Large Datasets
- **CPU**: 8+ vCPUs
- **RAM**: 32GB+
- **Storage**: 500GB+ NVMe SSD
- **GPU**: Optional, for faster embedding generation

### Installation on Ubuntu

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install system dependencies
sudo apt install gcc g++ curl

# Create application user
sudo useradd -m -s /bin/bash autopatternchecker
sudo usermod -aG sudo autopatternchecker

# Switch to application user
sudo su - autopatternchecker

# Clone repository
git clone <repository-url>
cd autopatternchecker

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Create directories
mkdir -p data output indices logs

# Set permissions
chmod 755 data output indices logs
```

### Systemd Service

Create `/etc/systemd/system/autopatternchecker.service`:

```ini
[Unit]
Description=AutoPatternChecker API Service
After=network.target

[Service]
Type=simple
User=autopatternchecker
Group=autopatternchecker
WorkingDirectory=/home/autopatternchecker/autopatternchecker
Environment=PATH=/home/autopatternchecker/autopatternchecker/venv/bin
Environment=PYTHONPATH=/home/autopatternchecker/autopatternchecker
ExecStart=/home/autopatternchecker/autopatternchecker/venv/bin/uvicorn autopatternchecker.api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable autopatternchecker
sudo systemctl start autopatternchecker
sudo systemctl status autopatternchecker
```

### Nginx Configuration

Create `/etc/nginx/sites-available/autopatternchecker`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/autopatternchecker /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL Configuration (Let's Encrypt)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Test renewal
sudo certbot renew --dry-run
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autopatternchecker

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: autopatternchecker-config
  namespace: autopatternchecker
data:
  config.yaml: |
    # Your configuration here
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autopatternchecker
  namespace: autopatternchecker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autopatternchecker
  template:
    metadata:
      labels:
        app: autopatternchecker
    spec:
      containers:
      - name: autopatternchecker
        image: autopatternchecker:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYTHONPATH
          value: /app
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: output-volume
          mountPath: /app/output
        - name: indices-volume
          mountPath: /app/indices
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: autopatternchecker-data
      - name: output-volume
        persistentVolumeClaim:
          claimName: autopatternchecker-output
      - name: indices-volume
        persistentVolumeClaim:
          claimName: autopatternchecker-indices
```

### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: autopatternchecker-service
  namespace: autopatternchecker
spec:
  selector:
    app: autopatternchecker
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autopatternchecker-ingress
  namespace: autopatternchecker
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: autopatternchecker-service
            port:
              number: 80
```

## Monitoring and Logging

### Prometheus Metrics

Add to your configuration:

```yaml
# monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'autopatternchecker'
      static_configs:
      - targets: ['autopatternchecker-service:80']
```

### Log Aggregation

```yaml
# fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/autopatternchecker/*.log
      pos_file /var/log/fluentd/autopatternchecker.log.pos
      tag autopatternchecker
      format json
    </source>
    
    <match autopatternchecker>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name autopatternchecker
    </match>
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/autopatternchecker"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup data
cp -r /app/data $BACKUP_DIR/$DATE/
cp -r /app/output $BACKUP_DIR/$DATE/
cp -r /app/indices $BACKUP_DIR/$DATE/

# Compress backup
tar -czf $BACKUP_DIR/autopatternchecker_$DATE.tar.gz -C $BACKUP_DIR $DATE

# Remove uncompressed backup
rm -rf $BACKUP_DIR/$DATE

# Keep only last 7 days of backups
find $BACKUP_DIR -name "autopatternchecker_*.tar.gz" -mtime +7 -delete

echo "Backup completed: autopatternchecker_$DATE.tar.gz"
```

### Recovery

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop service
sudo systemctl stop autopatternchecker

# Extract backup
tar -xzf $BACKUP_FILE -C /tmp

# Restore data
cp -r /tmp/*/data/* /app/data/
cp -r /tmp/*/output/* /app/output/
cp -r /tmp/*/indices/* /app/indices/

# Set permissions
chown -R autopatternchecker:autopatternchecker /app/data /app/output /app/indices

# Start service
sudo systemctl start autopatternchecker

echo "Recovery completed"
```

## Security Considerations

### API Security

1. **Authentication**: Implement API key authentication
2. **Rate Limiting**: Use nginx or API gateway for rate limiting
3. **Input Validation**: Validate all input parameters
4. **HTTPS**: Always use HTTPS in production

### Data Security

1. **Encryption**: Encrypt sensitive data at rest
2. **Access Control**: Implement proper file permissions
3. **Audit Logging**: Log all API access and changes
4. **Backup Encryption**: Encrypt backup files

### Network Security

1. **Firewall**: Configure proper firewall rules
2. **VPN**: Use VPN for administrative access
3. **Network Segmentation**: Isolate the application network

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase container memory limits
2. **Slow Performance**: Check FAISS index configuration
3. **API Timeouts**: Increase timeout settings
4. **Disk Space**: Monitor disk usage and clean old logs

### Debugging

```bash
# Check service status
sudo systemctl status autopatternchecker

# View logs
sudo journalctl -u autopatternchecker -f

# Check API health
curl http://localhost:8000/health

# Check resource usage
docker stats autopatternchecker
```

### Performance Tuning

1. **FAISS Index**: Use appropriate index type for your data size
2. **Embedding Model**: Choose model based on speed vs accuracy tradeoff
3. **Batch Size**: Optimize embedding batch size
4. **Caching**: Implement Redis caching for frequently accessed data