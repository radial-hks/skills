# FastAPI Dependencies and Middleware Reference

This document provides comprehensive guidance on implementing dependencies and middleware in FastAPI applications, covering authentication, authorization, logging, rate limiting, and custom middleware patterns.

## Table of Contents

1. [Dependencies Overview](#dependencies-overview)
2. [Authentication Dependencies](#authentication-dependencies)
3. [Authorization Dependencies](#authorization-dependencies)
4. [Database Dependencies](#database-dependencies)
5. [Utility Dependencies](#utility-dependencies)
6. [Middleware Overview](#middleware-overview)
7. [Authentication Middleware](#authentication-middleware)
8. [Logging Middleware](#logging-middleware)
9. [Error Handling Middleware](#error-handling-middleware)
10. [Rate Limiting Middleware](#rate-limiting-middleware)
11. [Custom Middleware Patterns](#custom-middleware-patterns)
12. [Testing Dependencies and Middleware](#testing-dependencies-and-middleware)

## Dependencies Overview

### Basic Dependency Structure

```python
from fastapi import Depends, FastAPI, HTTPException
from typing import Optional

app = FastAPI()

# Simple dependency
def get_query_parameter(q: Optional[str] = None):
    """Extract query parameter."""
    return q

@app.get("/items/")
async def read_items(query: str = Depends(get_query_parameter)):
    return {"query": query}

# Dependency with parameters
def get_pagination_params(skip: int = 0, limit: int = 10):
    """Get pagination parameters."""
    return {"skip": skip, "limit": limit}

@app.get("/users/")
async def get_users(pagination: dict = Depends(get_pagination_params)):
    return pagination
```

### Class Dependencies

```python
from fastapi import Depends
from typing import Optional

class CommonQueryParams:
    """Common query parameters dependency."""
    
    def __init__(
        self,
        q: Optional[str] = None,
        skip: int = 0,
        limit: int = 10,
        sort_by: Optional[str] = None,
        order: str = "asc"
    ):
        self.q = q
        self.skip = skip
        self.limit = limit
        self.sort_by = sort_by
        self.order = order
    
    def validate(self):
        """Validate parameters."""
        if self.limit > 100:
            raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
        if self.order not in ["asc", "desc"]:
            raise HTTPException(status_code=400, detail="Order must be 'asc' or 'desc'")
        return self

@app.get("/items/")
async def get_items(params: CommonQueryParams = Depends(CommonQueryParams)):
    params.validate()
    return params
```

### Dependency Caching

```python
from functools import lru_cache

@lru_cache()
def get_cached_settings():
    """Get cached application settings."""
    return Settings()

async def get_settings_dependency():
    """Dependency that returns cached settings."""
    return get_cached_settings()

@app.get("/settings/")
async def get_settings(settings: Settings = Depends(get_settings_dependency)):
    return settings
```

## Authentication Dependencies

### JWT Token Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional

security = HTTPBearer()

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: list = []

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, user_id=user_id)
    except jwt.PyJWTError:
        raise credentials_exception
    
    return token_data

# Usage
@app.get("/users/me/")
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    return {"username": current_user.username, "user_id": current_user.user_id}
```

### API Key Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

API_KEYS = {
    "test-api-key-1": {"user_id": 1, "name": "Test User 1"},
    "test-api-key-2": {"user_id": 2, "name": "Test User 2"},
}

async def get_api_key_user(api_key: str = Depends(api_key_header)):
    """Get user from API key."""
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return API_KEYS[api_key]

# Usage
@app.get("/api-key-data/")
async def get_api_key_data(user: dict = Depends(get_api_key_user)):
    return {"user": user, "message": "API key authenticated"}
```

### OAuth2 Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    """User model."""
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    """User model with password."""
    hashed_password: str

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Get password hash."""
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    # This would normally query the database
    fake_users_db = {
        "johndoe": {
            "username": "johndoe",
            "full_name": "John Doe",
            "email": "johndoe@example.com",
            "hashed_password": get_password_hash("secret"),
            "disabled": False,
        }
    }
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

async def get_current_user_from_oauth2(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from OAuth2 token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user

# Token endpoint
@app.post("/token/")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.username}, 
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}
```

## Authorization Dependencies

### Role-Based Authorization

```python
from typing import List

class UserWithRoles(User):
    """User with roles."""
    roles: List[str] = []

def require_role(allowed_roles: List[str]):
    """Dependency factory for role-based authorization."""
    async def role_checker(current_user: UserWithRoles = Depends(get_current_user)):
        if not any(role in current_user.roles for role in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker

# Usage
@app.get("/admin-only/")
async def admin_only_endpoint(
    current_user: UserWithRoles = Depends(require_role(["admin"]))
):
    return {"message": "Admin access granted", "user": current_user}

@app.get("/moderator-or-admin/")
async def moderator_or_admin_endpoint(
    current_user: UserWithRoles = Depends(require_role(["admin", "moderator"]))
):
    return {"message": "Moderator/Admin access granted", "user": current_user}
```

### Permission-Based Authorization

```python
class PermissionChecker:
    """Permission-based authorization dependency."""
    
    def __init__(self, required_permissions: List[str]):
        self.required_permissions = required_permissions
    
    async def __call__(self, current_user: UserWithRoles = Depends(get_current_user)):
        user_permissions = set()
        
        # Map roles to permissions
        role_permissions = {
            "admin": ["read", "write", "delete", "admin"],
            "moderator": ["read", "write", "moderate"],
            "user": ["read", "write"],
            "guest": ["read"]
        }
        
        for role in current_user.roles:
            user_permissions.update(role_permissions.get(role, []))
        
        if not all(permission in user_permissions for permission in self.required_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {self.required_permissions}"
            )
        
        return current_user

# Usage
@app.get("/write-required/")
async def write_required_endpoint(
    current_user: UserWithRoles = Depends(PermissionChecker(["write"]))
):
    return {"message": "Write permission granted", "user": current_user}

@app.delete("/delete-required/")
async def delete_required_endpoint(
    current_user: UserWithRoles = Depends(PermissionChecker(["delete"]))
):
    return {"message": "Delete permission granted", "user": current_user}
```

## Database Dependencies

### Async Database Session

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

# Database configuration
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/db"

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db_session() -> AsyncSession:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Usage with automatic transaction management
@asynccontextmanager
async def get_db_transaction():
    """Get database session with transaction management."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

@app.get("/users/{user_id}/")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """Get user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### Database Connection Health Check

```python
from sqlalchemy import text
import time

class DatabaseHealth:
    """Database health check dependency."""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
    
    async def __call__(self, db: AsyncSession = Depends(get_db_session)):
        """Check database health."""
        start_time = time.time()
        try:
            # Simple query to check connection
            result = await db.execute(text("SELECT 1"))
            result.scalar()
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "timeout": self.timeout
            }
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Database health check failed: {str(e)}"
            )

# Usage
@app.get("/health/database/")
async def check_database_health(health: dict = Depends(DatabaseHealth(timeout=3.0))):
    return health
```

## Utility Dependencies

### Request ID Generation

```python
import uuid
from fastapi import Request

class RequestID:
    """Generate and track request IDs."""
    
    def __init__(self, request: Request):
        self.request = request
        self.request_id = str(uuid.uuid4())
    
    def get_request_id(self) -> str:
        """Get request ID."""
        # Check if request ID already exists in headers
        existing_id = self.request.headers.get("X-Request-ID")
        if existing_id:
            return existing_id
        return self.request_id

async def get_request_id(request: Request) -> str:
    """Get request ID dependency."""
    request_id_service = RequestID(request)
    return request_id_service.get_request_id()

# Usage
@app.get("/request-id/")
async def get_request_id_endpoint(request_id: str = Depends(get_request_id)):
    return {"request_id": request_id}
```

### Rate Limiting Dependency

```python
import time
from collections import defaultdict
from typing import Dict

class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, calls: int = 10, period: int = 60):
        self.calls = calls
        self.period = period
        self.users: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to make request."""
        now = time.time()
        
        # Clean old entries
        self.users[user_id] = [
            timestamp for timestamp in self.users[user_id]
            if now - timestamp < self.period
        ]
        
        # Check if under limit
        if len(self.users[user_id]) < self.calls:
            self.users[user_id].append(now)
            return True
        
        return False
    
    def get_retry_after(self, user_id: str) -> int:
        """Get retry after time in seconds."""
        if not self.users[user_id]:
            return 0
        
        oldest_request = min(self.users[user_id])
        retry_after = self.period - (time.time() - oldest_request)
        return max(0, int(retry_after))

# Global rate limiter instance
rate_limiter = RateLimiter(calls=10, period=60)

def get_rate_limiter():
    """Get rate limiter dependency."""
    return rate_limiter

async def rate_limit_check(
    user_id: str = Depends(get_current_user),
    limiter: RateLimiter = Depends(get_rate_limiter)
):
    """Rate limiting dependency."""
    if not limiter.is_allowed(user_id.username):
        retry_after = limiter.get_retry_after(user_id.username)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )
    return user_id

# Usage
@app.get("/rate-limited/")
async def rate_limited_endpoint(
    current_user: User = Depends(rate_limit_check)
):
    return {"message": "Rate limited endpoint", "user": current_user}
```

## Middleware Overview

### Basic Middleware Structure

```python
from fastapi import FastAPI, Request, Response
import time

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### Middleware Order

```python
# Middleware is applied in the order it's defined
@app.middleware("http")
async def first_middleware(request: Request, call_next):
    print("First middleware - before")
    response = await call_next(request)
    print("First middleware - after")
    return response

@app.middleware("http")
async def second_middleware(request: Request, call_next):
    print("Second middleware - before")
    response = await call_next(request)
    print("Second middleware - after")
    return response
```

## Authentication Middleware

### JWT Authentication Middleware

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import jwt

class JWTAuthMiddleware:
    """JWT Authentication middleware."""
    
    def __init__(self, app: FastAPI, secret_key: str, algorithm: str = "HS256"):
        self.app = app
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    async def __call__(self, request: Request, call_next):
        """Process JWT authentication."""
        # Skip authentication for public endpoints
        if request.url.path in ["/", "/docs", "/openapi.json", "/health"]:
            response = await call_next(request)
            return response
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid authorization header"}
            )
        
        token = auth_header.split(" ")[1]
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            request.state.user = payload
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has expired"}
            )
        except jwt.JWTError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid token"}
            )
        
        response = await call_next(request)
        return response

# Usage
app = FastAPI()
jwt_middleware = JWTAuthMiddleware(app, SECRET_KEY)
app.add_middleware(BaseHTTPMiddleware, dispatch=jwt_middleware)
```

### Multi-Authentication Middleware

```python
class MultiAuthMiddleware:
    """Middleware supporting multiple authentication methods."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """Process multiple authentication methods."""
        # Skip authentication for public endpoints
        if request.url.path in ["/", "/docs", "/openapi.json", "/health"]:
            response = await call_next(request)
            return response
        
        # Try JWT authentication first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                request.state.user = payload
                request.state.auth_method = "jwt"
            except jwt.JWTError:
                pass
        
        # Try API key authentication
        if not hasattr(request.state, 'user'):
            api_key = request.headers.get("X-API-Key")
            if api_key and api_key in API_KEYS:
                request.state.user = API_KEYS[api_key]
                request.state.auth_method = "api_key"
        
        # Check if authentication was successful
        if not hasattr(request.state, 'user'):
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"}
            )
        
        response = await call_next(request)
        return response
```

## Logging Middleware

### Request/Response Logging

```python
import logging
import json
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LoggingMiddleware:
    """Request and response logging middleware."""
    
    def __init__(self, app: FastAPI, log_level: str = "INFO"):
        self.app = app
        self.log_level = log_level
    
    async def __call__(self, request: Request, call_next):
        """Log request and response details."""
        # Request details
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()
        
        # Log request
        request_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        logger.log(getattr(logging, self.log_level), f"Request: {json.dumps(request_data)}")
        
        # Process request
        response = await call_next(request)
        
        # Response details
        process_time = time.time() - start_time
        response_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "process_time": process_time,
        }
        
        logger.log(getattr(logging, self.log_level), f"Response: {json.dumps(response_data)}")
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response

# Usage
app = FastAPI()
logging_middleware = LoggingMiddleware(app)
app.add_middleware(BaseHTTPMiddleware, dispatch=logging_middleware)
```

### Structured Logging Middleware

```python
from pythonjsonlogger import jsonlogger
import logging

class StructuredLoggingMiddleware:
    """Structured JSON logging middleware."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.logger = logging.getLogger(__name__)
        
        # Configure JSON formatter
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        logHandler.setFormatter(formatter)
        self.logger.addHandler(logHandler)
        self.logger.setLevel(logging.INFO)
    
    async def __call__(self, request: Request, call_next):
        """Log structured request and response data."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()
        
        # Log request
        self.logger.info("request", extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": start_time
        })
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        self.logger.info("response", extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "process_time_ms": process_time * 1000,
            "timestamp": time.time()
        })
        
        response.headers["X-Request-ID"] = request_id
        return response
```

## Error Handling Middleware

### Global Error Handler

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

class ErrorHandlingMiddleware:
    """Global error handling middleware."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """Handle errors globally."""
        try:
            response = await call_next(request)
            return response
        except RequestValidationError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "validation_error",
                    "message": "Request validation failed",
                    "details": exc.errors()
                }
            )
        except StarletteHTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": "http_exception",
                    "message": exc.detail
                }
            )
        except Exception as exc:
            # Log the full exception for debugging
            logger.error(f"Unhandled exception: {str(exc)}")
            logger.error(traceback.format_exc())
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An internal server error occurred"
                }
            )

# Alternative approach using exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": exc.errors()
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_exception",
            "message": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An internal server error occurred"
        }
    )
```

### Custom Exception Classes

```python
class BusinessLogicException(Exception):
    """Custom business logic exception."""
    
    def __init__(self, message: str, error_code: str = None, status_code: int = 400):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

class ResourceNotFoundException(BusinessLogicException):
    """Resource not found exception."""
    
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} with identifier {identifier} not found",
            error_code="RESOURCE_NOT_FOUND",
            status_code=404
        )

class InsufficientFundsException(BusinessLogicException):
    """Insufficient funds exception."""
    
    def __init__(self, required: float, available: float):
        super().__init__(
            message=f"Insufficient funds. Required: ${required}, Available: ${available}",
            error_code="INSUFFICIENT_FUNDS",
            status_code=400
        )

# Exception handler for custom exceptions
@app.exception_handler(BusinessLogicException)
async def business_logic_exception_handler(request: Request, exc: BusinessLogicException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message
        }
    )

# Usage in endpoints
@app.get("/users/{user_id}/")
async def get_user(user_id: int):
    user = await get_user_from_db(user_id)
    if not user:
        raise ResourceNotFoundException("User", user_id)
    return user
```

## Rate Limiting Middleware

### Token Bucket Rate Limiter

```python
import time
import threading
from collections import defaultdict
from typing import Dict

class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from the bucket."""
        with self.lock:
            now = time.time()
            # Refill tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get wait time for tokens to be available."""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            return (tokens - self.tokens) / self.refill_rate

class TokenBucketRateLimitMiddleware:
    """Token bucket rate limiting middleware."""
    
    def __init__(
        self,
        app: FastAPI,
        default_capacity: int = 100,
        default_refill_rate: float = 10.0,
        key_func=None
    ):
        self.app = app
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self.key_func = key_func or self._default_key_func
        self.buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(default_capacity, default_refill_rate)
        )
    
    def _default_key_func(self, request: Request) -> str:
        """Default key function using client IP."""
        return request.client.host if request.client else "unknown"
    
    async def __call__(self, request: Request, call_next):
        """Apply rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health"]:
            response = await call_next(request)
            return response
        
        key = self.key_func(request)
        bucket = self.buckets[key]
        
        if not bucket.consume():
            wait_time = bucket.get_wait_time()
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Rate limit exceeded",
                    "retry_after": wait_time
                },
                headers={"Retry-After": str(int(wait_time))}
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.default_capacity)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + bucket.get_wait_time(1)))
        
        return response

# Usage
app = FastAPI()
rate_limit_middleware = TokenBucketRateLimitMiddleware(app)
app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware)
```

### Redis-Based Rate Limiting

```python
import redis
import json
import time

class RedisRateLimitMiddleware:
    """Redis-based rate limiting middleware."""
    
    def __init__(
        self,
        app: FastAPI,
        redis_client: redis.Redis,
        requests_per_minute: int = 60,
        key_prefix: str = "rate_limit"
    ):
        self.app = app
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
        self.key_prefix = key_prefix
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """Get rate limit key for request."""
        client_ip = request.client.host if request.client else "unknown"
        return f"{self.key_prefix}:{client_ip}"
    
    async def __call__(self, request: Request, call_next):
        """Apply Redis-based rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health"]:
            response = await call_next(request)
            return response
        
        key = self._get_rate_limit_key(request)
        now = time.time()
        
        # Get current request count
        current_count = self.redis.get(key)
        if current_count is None:
            current_count = 0
        else:
            current_count = int(current_count)
        
        # Check if limit exceeded
        if current_count >= self.requests_per_minute:
            ttl = self.redis.ttl(key)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Rate limit exceeded",
                    "retry_after": ttl
                },
                headers={"Retry-After": str(ttl)}
            )
        
        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)  # 1 minute TTL
        pipe.execute()
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.requests_per_minute - current_count - 1
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        
        return response
```

## Custom Middleware Patterns

### Request/Response Transformation

```python
class RequestTransformMiddleware:
    """Transform request and response data."""
    
    def __init__(self, app: FastAPI, transform_request: bool = True, transform_response: bool = True):
        self.app = app
        self.transform_request = transform_request
        self.transform_response = transform_response
    
    async def __call__(self, request: Request, call_next):
        """Transform request and response."""
        if self.transform_request:
            # Transform request body
            body = await request.body()
            if body:
                try:
                    data = json.loads(body)
                    # Add timestamp to request data
                    data["_timestamp"] = time.time()
                    
                    # Create new request with transformed body
                    new_body = json.dumps(data).encode()
                    request._body = new_body
                except json.JSONDecodeError:
                    pass
        
        response = await call_next(request)
        
        if self.transform_response:
            # Transform response headers
            response.headers["X-Transformed"] = "true"
            response.headers["X-Timestamp"] = str(time.time())
        
        return response
```

### CORS Middleware Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com", "https://app.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
    max_age=3600,  # 1 hour
)
```

### Gzip Compression Middleware

```python
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### HTTPS Redirect Middleware

```python
class HTTPSRedirectMiddleware:
    """Redirect HTTP requests to HTTPS."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """Redirect to HTTPS if request is HTTP."""
        # Check if request is secure
        if request.url.scheme == "http":
            # Build HTTPS URL
            https_url = request.url.replace(scheme="https")
            return JSONResponse(
                status_code=301,
                headers={"Location": str(https_url)},
                content={"message": "Redirecting to HTTPS"}
            )
        
        response = await call_next(request)
        return response
```

## Testing Dependencies and Middleware

### Testing Dependencies

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

@pytest.fixture
def mock_current_user():
    """Mock current user for testing."""
    return User(username="testuser", email="test@example.com", roles=["user"])

@pytest.fixture
def client_with_mock_auth(mock_current_user):
    """Test client with mocked authentication."""
    with patch("__main__.get_current_user", return_value=mock_current_user):
        client = TestClient(app)
        yield client

def test_protected_endpoint_with_mock_auth(client_with_mock_auth):
    """Test protected endpoint with mocked authentication."""
    response = client_with_mock_auth.get("/users/me/")
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

def test_protected_endpoint_with_real_auth():
    """Test protected endpoint with real authentication."""
    client = TestClient(app)
    
    # First, get a valid token
    response = client.post(
        "/token/",
        data={"username": "testuser", "password": "testpass"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # Use token to access protected endpoint
    response = client.get(
        "/users/me/",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
```

### Testing Middleware

```python
def test_middleware_headers():
    """Test middleware adds correct headers."""
    client = TestClient(app)
    response = client.get("/")
    
    # Check that middleware added headers
    assert "X-Process-Time" in response.headers
    assert "X-Request-ID" in response.headers
    assert float(response.headers["X-Process-Time"]) > 0

def test_rate_limiting_middleware():
    """Test rate limiting middleware."""
    client = TestClient(app)
    
    # Make multiple requests to trigger rate limiting
    for i in range(15):  # Assuming limit is 10 per minute
        response = client.get("/rate-limited/")
        if i >= 10:
            assert response.status_code == 429
            assert "Retry-After" in response.headers
        else:
            assert response.status_code == 200

def test_error_handling_middleware():
    """Test error handling middleware."""
    client = TestClient(app)
    
    # Test validation error
    response = client.post("/users/", json={"invalid": "data"})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    
    # Test business logic error
    response = client.get("/users/999/")  # Non-existent user
    assert response.status_code == 404
    assert response.json()["error"] == "RESOURCE_NOT_FOUND"
```

## Best Practices

1. **Keep dependencies focused**: Each dependency should have a single responsibility
2. **Use dependency caching**: Cache expensive operations when possible
3. **Handle errors gracefully**: Provide meaningful error messages in dependencies
4. **Test dependencies thoroughly**: Write comprehensive tests for all dependencies
5. **Use middleware sparingly**: Only use middleware for cross-cutting concerns
6. **Order middleware correctly**: Place authentication before authorization, logging first
7. **Monitor performance**: Track middleware performance impact
8. **Document dependencies**: Clearly document dependency requirements and behavior
9. **Use type hints**: Provide proper type hints for better IDE support
10. **Handle async operations**: Ensure middleware properly handles async operations