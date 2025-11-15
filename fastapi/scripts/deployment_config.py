#!/usr/bin/env python3
"""
FastAPI Deployment Configuration Script

This script provides deployment configurations for FastAPI applications,
including Docker, Kubernetes, AWS, Google Cloud, and production settings.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

class FastAPIDeploymentGenerator:
    """Generate deployment configurations for FastAPI applications."""
    
    def __init__(self):
        """Initialize deployment generator."""
        self.docker_base_image = "python:3.11-slim"
        self.fastapi_port = 8000
        self.default_workers = 4
    
    def generate_dockerfile(self, project_name: str) -> str:
        """Generate Dockerfile for FastAPI application."""
        
        code = f'''# FastAPI Dockerfile for {project_name}
FROM {self.docker_base_image}

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE {self.fastapi_port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.fastapi_port}/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{self.fastapi_port}", "--workers", "{self.default_workers}"]
'''
        return code
    
    def generate_docker_compose(self, project_name: str) -> str:
        """Generate Docker Compose configuration."""
        
        code = f'''# Docker Compose for {project_name}
version: '3.8'

services:
  app:
    build: .
    ports:
      - "{self.fastapi_port}:{self.fastapi_port}"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/{project_name.lower()}_db
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=your-secret-key-here
      - ENVIRONMENT=production
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self.fastapi_port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB={project_name.lower()}_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d {project_name.lower()}_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    driver: bridge
'''
        return code
    
    def generate_nginx_config(self, project_name: str) -> str:
        """Generate Nginx configuration."""
        
        code = f'''# Nginx configuration for {project_name}
events {{
    worker_connections 1024;
}}

http {{
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;

    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;

    # Upstream configuration
    upstream {project_name.lower()}_app {{
        server app:{self.fastapi_port};
        keepalive 32;
    }}

    # HTTP server
    server {{
        listen 80;
        server_name localhost;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }}

    # HTTPS server
    server {{
        listen 443 ssl http2;
        server_name localhost;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Client max body size
        client_max_body_size 10M;

        # Health check endpoint
        location /health {{
            access_log off;
            proxy_pass http://{project_name.lower()}_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # API endpoints with rate limiting
        location /api/ {{
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://{project_name.lower()}_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }}

        # Auth endpoints with stricter rate limiting
        location /auth/ {{
            limit_req zone=auth burst=5 nodelay;
            
            proxy_pass http://{project_name.lower()}_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # Static files
        location /static/ {{
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }}

        # Default location
        location / {{
            proxy_pass http://{project_name.lower()}_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
    }}
}}
'''
        return code
    
    def generate_kubernetes_deployment(self, project_name: str) -> str:
        """Generate Kubernetes deployment configuration."""
        
        code = f'''# Kubernetes Deployment for {project_name}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {project_name.lower()}-deployment
  labels:
    app: {project_name.lower()}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {project_name.lower()}
  template:
    metadata:
      labels:
        app: {project_name.lower()}
    spec:
      containers:
      - name: {project_name.lower()}
        image: {project_name.lower()}:latest
        ports:
        - containerPort: {self.fastapi_port}
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: {project_name.lower()}-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: {project_name.lower()}-secrets
              key: secret-key
        - name: REDIS_URL
          value: redis://redis-service:6379/0
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: {self.fastapi_port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {self.fastapi_port}
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {project_name.lower()}-service
spec:
  selector:
    app: {project_name.lower()}
  ports:
  - protocol: TCP
    port: 80
    targetPort: {self.fastapi_port}
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {project_name.lower()}-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "10"
spec:
  tls:
  - hosts:
    - {project_name.lower()}.yourdomain.com
    secretName: {project_name.lower()}-tls
  rules:
  - host: {project_name.lower()}.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {project_name.lower()}-service
            port:
              number: 80
---
apiVersion: v1
kind: Secret
metadata:
  name: {project_name.lower()}-secrets
type: Opaque
data:
  database-url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc3dvcmRAcG9zdGdyZXM6NTQzMi97cHJvamVjdF9uYW1lLmxvd2VyKCl9X2Ri  # base64 encoded
  secret-key: eW91ci1zZWNyZXQta2V5LWhlcmU=  # base64 encoded
'''
        return code
    
    def generate_aws_ecs_config(self, project_name: str) -> str:
        """Generate AWS ECS configuration."""
        
        code = f'''# AWS ECS Task Definition for {project_name}
{{
  "family": "{project_name.lower()}-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskRole",
  "containerDefinitions": [
    {{
      "name": "{project_name.lower()}-container",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/{project_name.lower()}:latest",
      "portMappings": [
        {{
          "containerPort": {self.fastapi_port},
          "protocol": "tcp"
        }}
      ],
      "environment": [
        {{
          "name": "ENVIRONMENT",
          "value": "production"
        }},
        {{
          "name": "DATABASE_URL",
          "value": "postgresql://user:password@your-rds-endpoint:5432/{project_name.lower()}_db"
        }},
        {{
          "name": "SECRET_KEY",
          "value": "your-secret-key-here"
        }},
        {{
          "name": "REDIS_URL",
          "value": "redis://your-elasticache-endpoint:6379/0"
        }}
      ],
      "secrets": [
        {{
          "name": "DATABASE_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:YOUR_REGION:YOUR_ACCOUNT_ID:secret:{project_name.lower()}/database-password"
        }}
      ],
      "logConfiguration": {{
        "logDriver": "awslogs",
        "options": {{
          "awslogs-group": "/ecs/{project_name.lower()}",
          "awslogs-region": "YOUR_REGION",
          "awslogs-stream-prefix": "ecs"
        }}
      }},
      "healthCheck": {{
        "command": ["CMD-SHELL", "curl -f http://localhost:{self.fastapi_port}/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }}
    }}
  ]
}}
'''
        return code
    
    def generate_google_cloud_run_config(self, project_name: str) -> str:
        """Generate Google Cloud Run configuration."""
        
        code = f'''# Google Cloud Run Configuration for {project_name}
# Save this as cloudbuild.yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/{project_name.lower()}:$COMMIT_SHA', '.']
  
  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/{project_name.lower()}:$COMMIT_SHA']
  
  # Deploy container image to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - '{project_name.lower()}'
      - '--image'
      - 'gcr.io/$PROJECT_ID/{project_name.lower()}:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '512Mi'
      - '--cpu'
      - '1'
      - '--max-instances'
      - '10'
      - '--min-instances'
      - '0'
      - '--timeout'
      - '300s'
      - '--concurrency'
      - '80'
      - '--port'
      - '{self.fastapi_port}'
      - '--set-env-vars'
      - 'ENVIRONMENT=production,DATABASE_URL=postgresql://user:password@your-cloudsql-instance:5432/{project_name.lower()}_db,SECRET_KEY=your-secret-key-here'
      - '--set-secrets'
      - 'DATABASE_PASSWORD={project_name.lower()}-db-password:latest'

# Configuration
options:
  logging: CLOUD_LOGGING_ONLY

# Trigger configuration
triggers:
  - name: '{project_name.lower()}-trigger'
    description: 'Build and deploy {project_name.lower()}'
    github:
      name: 'your-repo-name'
      owner: 'your-github-username'
      push:
        branch: '^main$'

# Substitutions
substitutions:
  _SERVICE_NAME: '{project_name.lower()}'
  _REGION: 'us-central1'
  _MEMORY: '512Mi'
  _CPU: '1'
'''
        return code
    
    def generate_heroku_config(self, project_name: str) -> str:
        """Generate Heroku configuration."""
        
        code = f'''# Heroku Configuration for {project_name}
# Save this as app.json for Heroku deployment
{{
  "name": "{project_name}",
  "description": "FastAPI application deployed to Heroku",
  "repository": "https://github.com/your-username/{project_name.lower()}",
  "logo": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png",
  "keywords": ["python", "fastapi", "api", "rest"],
  "stack": "heroku-20",
  "buildpacks": [
    {{
      "url": "heroku/python"
    }}
  ],
  "env": {{
    "ENVIRONMENT": {{
      "description": "Application environment",
      "value": "production"
    }},
    "SECRET_KEY": {{
      "description": "Secret key for the application",
      "generator": "secret"
    }},
    "DATABASE_URL": {{
      "description": "Database connection URL",
      "value": "postgresql://user:password@host:5432/{project_name.lower()}_db"
    }},
    "REDIS_URL": {{
      "description": "Redis connection URL",
      "value": "redis://host:6379/0"
    }},
    "ALLOWED_HOSTS": {{
      "description": "Allowed hosts for the application",
      "value": "*"
    }},
    "CORS_ORIGINS": {{
      "description": "CORS origins",
      "value": "https://{project_name.lower()}.herokuapp.com"
    }},
    "PORT": {{
      "description": "Port to run the application on",
      "value": "{self.fastapi_port}"
    }}
  }},
  "formation": {{
    "web": {{
      "quantity": 1,
      "size": "basic"
    }}
  }},
  "addons": [
    "heroku-postgresql:mini",
    "heroku-redis:mini"
  ],
  "scripts": {{
    "postdeploy": "python -m alembic upgrade head"
  }}
}}

# Procfile for Heroku
web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2
worker: celery -A celery_app worker --loglevel=info
beat: celery -A celery_app beat --loglevel=info
'''
        return code
    
    def generate_production_config(self, project_name: str) -> str:
        """Generate production configuration settings."""
        
        code = f'''# Production Configuration for {project_name}
"""
Production settings for FastAPI application.
This module contains all production-specific configurations.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator

class ProductionSettings(BaseSettings):
    """Production settings for FastAPI application."""
    
    # Application settings
    APP_NAME: str = "{project_name}"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    TESTING: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = {self.fastapi_port}
    WORKERS: int = 4
    RELOAD: bool = False
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = ["https://yourdomain.com"]
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/{project_name.lower()}_db")
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    
    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_POOL_SIZE: int = 10
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "/var/log/{project_name.lower()}/app.log"
    LOG_MAX_SIZE: int = 100 * 1024 * 1024  # 100MB
    LOG_BACKUP_COUNT: int = 5
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    
    # JWT settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    API_TITLE: str = "{project_name} API"
    API_DESCRIPTION: str = "Production API for {project_name}"
    API_VERSION: str = "1.0.0"
    
    # File upload settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "/app/uploads"
    ALLOWED_FILE_TYPES: List[str] = ["image/jpeg", "image/png", "image/gif", "application/pdf"]
    
    # Email settings
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_USE_TLS: bool = True
    
    # Monitoring settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_PATH: str = "/health"
    
    # Cache settings
    CACHE_TTL: int = 300  # 5 minutes
    CACHE_MAX_SIZE: int = 1000
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from comma-separated string."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True

# Production logging configuration
LOGGING_CONFIG = {{
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {{
        "json": {{
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }},
        "standard": {{
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }}
    }},
    "handlers": {{
        "console": {{
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }},
        "file": {{
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "/var/log/{project_name.lower()}/app.log",
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 5
        }},
        "error_file": {{
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "json",
            "filename": "/var/log/{project_name.lower()}/error.log",
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 5
        }}
    }},
    "loggers": {{
        "": {{
            "handlers": ["console", "file", "error_file"],
            "level": "INFO",
            "propagate": False
        }},
        "uvicorn": {{
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }},
        "fastapi": {{
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }}
    }}
}}

# Production security headers
SECURITY_HEADERS = {{
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self' https:;"
}}
'''
        return code
    
    def generate_requirements_txt(self, project_name: str) -> str:
        """Generate production requirements.txt."""
        
        code = f'''# Production requirements for {project_name}
# Core FastAPI dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database dependencies
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9
redis==5.0.1

# Security dependencies
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Async dependencies
asyncpg==0.29.0
aioredis==2.0.1

# Utility dependencies
python-dotenv==1.0.0
httpx==0.25.2
celery==5.3.4

# Logging dependencies
python-json-logger==2.0.7

# Monitoring dependencies
prometheus-client==0.19.0

# Production dependencies
gunicorn==21.2.0
supervisor==4.2.5

# Optional dependencies for enhanced features
# Uncomment as needed:
# slowapi==0.1.9  # Rate limiting
# pillow==10.1.0  # Image processing
# boto3==1.34.0  # AWS integration
# google-cloud-storage==2.12.0  # Google Cloud Storage
# azure-storage-blob==12.19.0  # Azure Blob Storage
'''
        return code

def main():
    """CLI interface for deployment configuration."""
    parser = argparse.ArgumentParser(description="Generate FastAPI deployment configurations")
    parser.add_argument("--project-name", default="MyFastAPI", help="Project name")
    parser.add_argument("--output-dir", default="deployment", help="Output directory")
    parser.add_argument("--docker", action="store_true", help="Generate Docker configuration")
    parser.add_argument("--kubernetes", action="store_true", help="Generate Kubernetes configuration")
    parser.add_argument("--aws", action="store_true", help="Generate AWS ECS configuration")
    parser.add_argument("--gcp", action="store_true", help="Generate Google Cloud Run configuration")
    parser.add_argument("--heroku", action="store_true", help="Generate Heroku configuration")
    parser.add_argument("--nginx", action="store_true", help="Generate Nginx configuration")
    parser.add_argument("--production", action="store_true", help="Generate production configuration")
    parser.add_argument("--requirements", action="store_true", help="Generate production requirements.txt")
    parser.add_argument("--all", action="store_true", help="Generate all deployment configurations")
    
    args = parser.parse_args()
    
    generator = FastAPIDeploymentGenerator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.all or args.docker:
        docker_dir = output_dir / "docker"
        docker_dir.mkdir(exist_ok=True)
        
        dockerfile = docker_dir / "Dockerfile"
        dockerfile.write_text(generator.generate_dockerfile(args.project_name))
        print(f"Dockerfile generated: {dockerfile}")
        
        docker_compose = docker_dir / "docker-compose.yml"
        docker_compose.write_text(generator.generate_docker_compose(args.project_name))
        print(f"Docker Compose configuration generated: {docker_compose}")
    
    if args.all or args.kubernetes:
        k8s_dir = output_dir / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        k8s_config = k8s_dir / "deployment.yaml"
        k8s_config.write_text(generator.generate_kubernetes_deployment(args.project_name))
        print(f"Kubernetes configuration generated: {k8s_config}")
    
    if args.all or args.aws:
        aws_dir = output_dir / "aws"
        aws_dir.mkdir(exist_ok=True)
        
        ecs_config = aws_dir / "ecs-task-definition.json"
        ecs_config.write_text(generator.generate_aws_ecs_config(args.project_name))
        print(f"AWS ECS configuration generated: {ecs_config}")
    
    if args.all or args.gcp:
        gcp_dir = output_dir / "gcp"
        gcp_dir.mkdir(exist_ok=True)
        
        cloudbuild_config = gcp_dir / "cloudbuild.yaml"
        cloudbuild_config.write_text(generator.generate_google_cloud_run_config(args.project_name))
        print(f"Google Cloud Run configuration generated: {cloudbuild_config}")
    
    if args.all or args.heroku:
        heroku_config = output_dir / "app.json"
        heroku_config.write_text(generator.generate_heroku_config(args.project_name))
        print(f"Heroku configuration generated: {heroku_config}")
        
        procfile = output_dir / "Procfile"
        procfile.write_text(generator.generate_heroku_config(args.project_name).split("# Procfile for Heroku")[1])
        print(f"Procfile generated: {procfile}")
    
    if args.all or args.nginx:
        nginx_dir = output_dir / "nginx"
        nginx_dir.mkdir(exist_ok=True)
        
        nginx_config = nginx_dir / "nginx.conf"
        nginx_config.write_text(generator.generate_nginx_config(args.project_name))
        print(f"Nginx configuration generated: {nginx_config}")
    
    if args.all or args.production:
        config_dir = output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        prod_config = config_dir / "production.py"
        prod_config.write_text(generator.generate_production_config(args.project_name))
        print(f"Production configuration generated: {prod_config}")
    
    if args.all or args.requirements:
        requirements = output_dir / "requirements.txt"
        requirements.write_text(generator.generate_requirements_txt(args.project_name))
        print(f"Production requirements generated: {requirements}")
    
    # Create deployment documentation
    if args.all:
        deployment_docs = output_dir / "README.md"
        docs_content = f'''# Deployment Guide for {args.project_name}

This directory contains deployment configurations for various platforms.

## Docker Deployment

```bash
# Build and run with Docker Compose
cd docker
docker-compose up -d

# Build and run with Docker
docker build -t {args.project_name.lower()} .
docker run -p {generator.fastapi_port}:{generator.fastapi_port} {args.project_name.lower()}
```

## Kubernetes Deployment

```bash
# Apply Kubernetes configurations
kubectl apply -f kubernetes/deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get services
kubectl get ingress
```

## AWS ECS Deployment

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json

# Create service
aws ecs create-service --cluster your-cluster --service-name {args.project_name.lower()}-service --task-definition {args.project_name.lower()}-task
```

## Google Cloud Run Deployment

```bash
# Submit build
gcloud builds submit --config=gcp/cloudbuild.yaml

# Or deploy directly
gcloud run deploy {args.project_name.lower()} --source . --region us-central1
```

## Heroku Deployment

```bash
# Create Heroku app
heroku create {args.project_name.lower()}

# Set environment variables
heroku config:set ENVIRONMENT=production
heroku config:set SECRET_KEY=your-secret-key

# Deploy
git push heroku main
```

## Production Configuration

The production configuration includes:
- Security headers and CORS settings
- Logging configuration with rotation
- Rate limiting and monitoring
- Database connection pooling
- Error handling and health checks

## Environment Variables

Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `SECRET_KEY`: Application secret key
- `REDIS_URL`: Redis connection string
- `ENVIRONMENT`: Application environment (production/development)

## Security Considerations

- Use HTTPS in production
- Implement proper authentication and authorization
- Configure rate limiting
- Use environment variables for sensitive data
- Enable logging and monitoring
- Regular security updates
'''
        deployment_docs.write_text(docs_content)
        print(f"Deployment documentation generated: {deployment_docs}")
    
    print("Deployment configuration completed!")

if __name__ == "__main__":
    main()