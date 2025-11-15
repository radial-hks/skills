#!/usr/bin/env python3
"""
FastAPI Security Setup Script

This script provides comprehensive security configurations for FastAPI applications,
including OAuth2, JWT tokens, API key authentication, rate limiting, and CORS setup.
"""

import os
import secrets
import argparse
from pathlib import Path
from typing import Dict, List, Optional

class FastAPISecuritySetup:
    """Generate security configurations for FastAPI applications."""
    
    def __init__(self):
        """Initialize security setup."""
        self.secret_key = secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def generate_oauth2_config(self, project_name: str) -> str:
        """Generate OAuth2 security configuration."""
        
        code = f'''"""
OAuth2 Security Configuration for {project_name}

This module provides OAuth2 authentication setup with JWT tokens.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# Security configuration
SECRET_KEY = "{self.secret_key}"
ALGORITHM = "{self.algorithm}"
ACCESS_TOKEN_EXPIRE_MINUTES = {self.access_token_expire_minutes}
REFRESH_TOKEN_EXPIRE_DAYS = {self.refresh_token_expire_days}

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list = []

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    scopes: list = []

class UserInDB(User):
    hashed_password: str

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({{"exp": expire, "type": "access"}})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({{"exp": expire, "type": "refresh"}})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={{"WWW-Authenticate": "Bearer"}},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # TODO: Implement user lookup from database
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database (placeholder)."""
    # TODO: Implement actual user lookup
    # This is a placeholder implementation
    fake_users_db = {{
        "admin": {{
            "username": "admin",
            "email": "admin@example.com",
            "full_name": "Admin User",
            "hashed_password": get_password_hash("secret"),
            "disabled": False,
            "scopes": ["read", "write", "admin"]
        }}
    }}
    
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None

async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

# Role-based dependency factories
def require_role(role: str):
    """Create dependency for specific role requirement."""
    def role_checker(current_user: User = Depends(get_current_active_user)):
        if role not in current_user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return current_user
    return role_checker

def require_scopes(required_scopes: list):
    """Create dependency for scope requirements."""
    def scope_checker(current_user: User = Depends(get_current_active_user)):
        user_scopes = set(current_user.scopes)
        required = set(required_scopes)
        if not required.issubset(user_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient scopes. Required: {{required_scopes}}"
            )
        return current_user
    return scope_checker
'''
        return code
    
    def generate_api_key_config(self) -> str:
        """Generate API key authentication configuration."""
        
        code = '''"""
API Key Authentication for FastAPI

This module provides API key authentication setup.
"""

from typing import Optional
from fastapi import HTTPException, status, Depends, Security
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# API Key configurations
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)

class APIKeyAuth(BaseModel):
    api_key: str
    name: str
    permissions: list = []

# Example API keys (use database in production)
API_KEYS = {{
    "demo-key-123": APIKeyAuth(
        api_key="demo-key-123",
        name="Demo Application",
        permissions=["read", "limited-write"]
    ),
    "admin-key-456": APIKeyAuth(
        api_key="admin-key-456",
        name="Admin Application",
        permissions=["read", "write", "admin"]
    )
}}

async def verify_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query)
) -> APIKeyAuth:
    """Verify API key from header or query parameter."""
    
    # Prefer header over query parameter
    api_key = api_key_header or api_key_query
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return API_KEYS[api_key]

async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Security(http_bearer)
) -> str:
    """Verify bearer token."""
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required"
        )
    
    token = credentials.credentials
    
    # TODO: Implement token verification logic
    # This could be JWT verification or database lookup
    
    if token != "valid-bearer-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token"
        )
    
    return token

def require_permission(permission: str):
    """Create dependency for specific permission requirement."""
    def permission_checker(api_key: APIKeyAuth = Depends(verify_api_key)):
        if permission not in api_key.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{{permission}}' required"
            )
        return api_key
    return permission_checker

# Combined authentication
def get_auth_dependency(use_oauth2: bool = True, use_api_key: bool = False):
    """Get appropriate authentication dependency."""
    if use_oauth2 and use_api_key:
        # Allow either OAuth2 or API key
        def combined_auth(
            current_user = Depends(get_current_active_user),
            api_key = Depends(verify_api_key)
        ):
            return {{"user": current_user, "api_key": api_key}}
        return combined_auth
    elif use_oauth2:
        return get_current_active_user
    elif use_api_key:
        return verify_api_key
    else:
        # No authentication required
        def no_auth():
            return None
        return no_auth
'''
        return code
    
    def generate_cors_config(self, allowed_origins: List[str] = None) -> str:
        """Generate CORS configuration."""
        
        if not allowed_origins:
            allowed_origins = ["http://localhost:3000", "http://localhost:8080"]
        
        origins_str = ', '.join([f'"{origin}"' for origin in allowed_origins])
        
        code = f'''"""
CORS Configuration for FastAPI

This module provides CORS middleware setup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# CORS configuration
ALLOWED_ORIGINS = [{origins_str}]
ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
ALLOWED_HEADERS = ["*"]
ALLOW_CREDENTIALS = True

def setup_cors(app: FastAPI):
    """Setup CORS middleware for FastAPI application."""
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=ALLOW_CREDENTIALS,
        allow_methods=ALLOWED_METHODS,
        allow_headers=ALLOWED_HEADERS,
    )
    
    return app

# CORS helper functions
def add_cors_headers(response, origin: str = None):
    """Add CORS headers to response."""
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = ", ".join(ALLOWED_METHODS)
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "3600"
    
    return response

def is_origin_allowed(origin: str) -> bool:
    """Check if origin is allowed."""
    if "*" in ALLOWED_ORIGINS:
        return True
    
    # Support wildcard subdomains
    for allowed in ALLOWED_ORIGINS:
        if "*" in allowed:
            # Convert wildcard to regex
            pattern = allowed.replace("*", ".*")
            import re
            if re.match(pattern, origin):
                return True
        elif allowed == origin:
            return True
    
    return False
'''
        return code
    
    def generate_rate_limiting_config(self) -> str:
        """Generate rate limiting configuration."""
        
        code = '''"""
Rate Limiting for FastAPI

This module provides rate limiting functionality.
"""

import time
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import HTTPException, status, Request
from fastapi.responses import JSONResponse

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limits."""
        now = time.time()
        
        # Clean old requests
        self._clean_old_requests(identifier, now)
        
        # Check minute limit
        minute_requests = [req for req in self.requests[identifier] if now - req < 60]
        if len(minute_requests) >= self.requests_per_minute:
            return False
        
        # Check hour limit
        hour_requests = [req for req in self.requests[identifier] if now - req < 3600]
        if len(hour_requests) >= self.requests_per_hour:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def _clean_old_requests(self, identifier: str, now: float):
        """Remove requests older than 1 hour."""
        while (self.requests[identifier] and 
               now - self.requests[identifier][0] > 3600):
            self.requests[identifier].popleft()
    
    def get_retry_after(self, identifier: str) -> float:
        """Get retry after time in seconds."""
        if not self.requests[identifier]:
            return 0
        
        now = time.time()
        oldest_request = self.requests[identifier][0]
        
        # Return time until oldest request is older than 1 minute
        retry_after = 60 - (now - oldest_request)
        return max(0, retry_after)

# Global rate limiter instance
rate_limiter = RateLimiter()

def get_client_identifier(request: Request) -> str:
    """Get client identifier from request."""
    # Try to get from header first
    client_id = request.headers.get("X-Client-ID")
    if client_id:
        return client_id
    
    # Fall back to IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    return request.client.host

def rate_limit(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    identifier_func = get_client_identifier
):
    """Rate limiting dependency."""
    
    def rate_limit_dependency(request: Request):
        # Create custom rate limiter if different limits
        if (requests_per_minute != rate_limiter.requests_per_minute or 
            requests_per_hour != rate_limiter.requests_per_hour):
            limiter = RateLimiter(requests_per_minute, requests_per_hour)
        else:
            limiter = rate_limiter
        
        identifier = identifier_func(request)
        
        if not limiter.is_allowed(identifier):
            retry_after = limiter.get_retry_after(identifier)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={{"Retry-After": str(int(retry_after))}}
            )
        
        return identifier
    
    return rate_limit_dependency

# Advanced rate limiting with Redis (optional)
class RedisRateLimiter:
    """Redis-based rate limiter for distributed systems."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            import redis
            self.redis = redis.from_url(redis_url)
            self.enabled = True
        except ImportError:
            print("Redis not available, falling back to in-memory rate limiter")
            self.enabled = False
            self.fallback = RateLimiter()
    
    def is_allowed(self, identifier: str, limit: int = 60, window: int = 60) -> bool:
        """Check if request is allowed using sliding window."""
        if not self.enabled:
            return self.fallback.is_allowed(identifier)
        
        key = f"rate_limit:{{identifier}}"
        now = int(time.time())
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Remove expired entries
        pipe.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {{str(now): now}})
        
        # Set expiration
        pipe.expire(key, window)
        
        results = pipe.execute()
        current_requests = results[1]
        
        return current_requests < limit
'''
        return code
    
    def generate_security_headers_config(self) -> str:
        """Generate security headers configuration."""
        
        code = '''"""
Security Headers for FastAPI

This module provides security headers middleware.
"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Dict, Any

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.config = {{
            "x_content_type_options": "nosniff",
            "x_frame_options": "DENY",
            "x_xss_protection": "1; mode=block",
            "strict_transport_security": "max-age=31536000; includeSubDomains",
            "referrer_policy": "strict-origin-when-cross-origin",
            "content_security_policy": "default-src 'self'",
            "permissions_policy": "geolocation=(), microphone=(), camera=()"
        }}
        self.config.update(kwargs)
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.config.items():
            if value:  # Only add if value is not empty
                header_name = header.replace("_", "-").title()
                response.headers[header_name] = value
        
        return response

def setup_security_headers(app, **kwargs):
    """Setup security headers middleware."""
    app.add_middleware(SecurityHeadersMiddleware, **kwargs)
    return app

# Security utilities
class SecurityConfig:
    """Security configuration."""
    
    def __init__(self):
        self.allowed_hosts = []
        self.blocked_user_agents = []
        self.max_content_length = 10 * 1024 * 1024  # 10MB
        self.allowed_content_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain"
        ]
    
    def is_host_allowed(self, host: str) -> bool:
        """Check if host is allowed."""
        if not self.allowed_hosts:
            return True
        
        return host in self.allowed_hosts
    
    def is_user_agent_blocked(self, user_agent: str) -> bool:
        """Check if user agent should be blocked."""
        for blocked in self.blocked_user_agents:
            if blocked.lower() in user_agent.lower():
                return True
        return False
    
    def is_content_type_allowed(self, content_type: str) -> bool:
        """Check if content type is allowed."""
        if not content_type:
            return True
        
        # Strip parameters
        main_type = content_type.split(";")[0].strip()
        return main_type in self.allowed_content_types

class RequestSecurityMiddleware(BaseHTTPMiddleware):
    """Validate incoming requests for security."""
    
    def __init__(self, app, config: SecurityConfig = None):
        super().__init__(app)
        self.config = config or SecurityConfig()
    
    async def dispatch(self, request: Request, call_next):
        # Check host
        host = request.headers.get("host", "")
        if not self.config.is_host_allowed(host):
            return JSONResponse(
                status_code=400,
                content={{"error": "Host not allowed"}}
            )
        
        # Check user agent
        user_agent = request.headers.get("user-agent", "")
        if self.config.is_user_agent_blocked(user_agent):
            return JSONResponse(
                status_code=400,
                content={{"error": "User agent blocked"}}
            )
        
        # Check content type
        content_type = request.headers.get("content-type", "")
        if not self.config.is_content_type_allowed(content_type):
            return JSONResponse(
                status_code=400,
                content={{"error": "Content type not allowed"}}
            )
        
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.config.max_content_length:
                    return JSONResponse(
                        status_code=413,
                        content={{"error": "Content too large"}}
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={{"error": "Invalid content length"}}
                )
        
        response = await call_next(request)
        return response

# Input validation utilities
def sanitize_input(value: str) -> str:
    """Sanitize input string."""
    if not isinstance(value, str):
        return value
    
    # Remove null bytes
    value = value.replace('\\0', '')
    
    # Strip whitespace
    value = value.strip()
    
    # Limit length
    max_length = 1000
    if len(value) > max_length:
        value = value[:max_length]
    
    return value

def validate_json_size(json_data: Dict[str, Any], max_size: int = 1024 * 1024) -> bool:
    """Validate JSON data size."""
    import json
    try:
        json_str = json.dumps(json_data)
        return len(json_str.encode('utf-8')) <= max_size
    except Exception:
        return False
'''
        return code
    
    def generate_security_config_file(self) -> str:
        """Generate main security configuration file."""
        
        code = f'''"""
Security Configuration for FastAPI Application

This module provides comprehensive security setup for FastAPI applications.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from .security.oauth2 import setup_oauth2
from .security.api_key import setup_api_key
from .security.cors import setup_cors
from .security.rate_limiting import setup_rate_limiting
from .security.headers import setup_security_headers

# Security settings
SECURITY_CONFIG = {{
    "secret_key": "{self.secret_key}",
    "algorithm": "{self.algorithm}",
    "access_token_expire_minutes": {self.access_token_expire_minutes},
    "refresh_token_expire_days": {self.refresh_token_expire_days},
    "rate_limit_per_minute": 60,
    "rate_limit_per_hour": 1000,
    "cors_origins": ["http://localhost:3000", "http://localhost:8080"],
    "allowed_hosts": ["*"],  # Configure for production
    "enable_swagger_auth": True
}}

def setup_security(app: FastAPI, config: dict = None) -> FastAPI:
    """Setup all security middleware and configurations."""
    
    config = config or SECURITY_CONFIG
    
    # Setup CORS
    app = setup_cors(app, origins=config.get("cors_origins", ["*"]))
    
    # Setup security headers
    app = setup_security_headers(app)
    
    # Setup rate limiting
    app = setup_rate_limiting(
        app,
        requests_per_minute=config.get("rate_limit_per_minute", 60),
        requests_per_hour=config.get("rate_limit_per_hour", 1000)
    )
    
    # Setup trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.get("allowed_hosts", ["*"])
    )
    
    # Setup authentication
    if config.get("enable_oauth2", True):
        app = setup_oauth2(app, config)
    
    if config.get("enable_api_key", False):
        app = setup_api_key(app, config)
    
    return app

# Security dependencies
from .security.oauth2 import get_current_active_user, require_role, require_scopes
from .security.api_key import verify_api_key, require_permission
from .security.rate_limiting import rate_limit

__all__ = [
    "setup_security",
    "get_current_active_user",
    "require_role",
    "require_scopes",
    "verify_api_key",
    "require_permission",
    "rate_limit",
    "SECURITY_CONFIG"
]
'''
        return code

def main():
    """CLI interface for security setup."""
    parser = argparse.ArgumentParser(description="Setup FastAPI security configurations")
    parser.add_argument("--project-name", default="MyFastAPI", help="Project name")
    parser.add_argument("--output-dir", default="security", help="Output directory")
    parser.add_argument("--oauth2", action="store_true", help="Generate OAuth2 configuration")
    parser.add_argument("--api-key", action="store_true", help="Generate API key configuration")
    parser.add_argument("--cors", action="store_true", help="Generate CORS configuration")
    parser.add_argument("--rate-limit", action="store_true", help="Generate rate limiting configuration")
    parser.add_argument("--headers", action="store_true", help="Generate security headers configuration")
    parser.add_argument("--all", action="store_true", help="Generate all security configurations")
    
    args = parser.parse_args()
    
    setup = FastAPISecuritySetup()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.all or args.oauth2:
        oauth2_dir = output_dir / "oauth2.py"
        oauth2_dir.write_text(setup.generate_oauth2_config(args.project_name))
        print(f"OAuth2 configuration generated: {{oauth2_dir}}")
    
    if args.all or args.api_key:
        api_key_file = output_dir / "api_key.py"
        api_key_file.write_text(setup.generate_api_key_config())
        print(f"API key configuration generated: {{api_key_file}}")
    
    if args.all or args.cors:
        cors_file = output_dir / "cors.py"
        cors_file.write_text(setup.generate_cors_config())
        print(f"CORS configuration generated: {{cors_file}}")
    
    if args.all or args.rate_limit:
        rate_limit_file = output_dir / "rate_limiting.py"
        rate_limit_file.write_text(setup.generate_rate_limiting_config())
        print(f"Rate limiting configuration generated: {{rate_limit_file}}")
    
    if args.all or args.headers:
        headers_file = output_dir / "headers.py"
        headers_file.write_text(setup.generate_security_headers_config())
        print(f"Security headers configuration generated: {{headers_file}}")
    
    if args.all:
        config_file = output_dir / "config.py"
        config_file.write_text(setup.generate_security_config_file())
        print(f"Main security configuration generated: {{config_file}}")
    
    print("Security setup completed!")

if __name__ == "__main__":
    main()