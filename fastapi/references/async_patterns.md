# FastAPI Async Patterns Reference

This document provides comprehensive guidance on implementing async patterns in FastAPI applications, covering best practices, common patterns, and performance optimization techniques.

## Table of Contents

1. [Async Fundamentals](#async-fundamentals)
2. [Database Async Patterns](#database-async-patterns)
3. [External API Integration](#external-api-integration)
4. [Background Tasks](#background-tasks)
5. [WebSocket Implementation](#websocket-implementation)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling](#error-handling)
8. [Testing Async Code](#testing-async-code)

## Async Fundamentals

### Basic Async Route Handlers

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/async-endpoint")
async def async_endpoint():
    """Basic async endpoint."""
    await asyncio.sleep(1)  # Simulate async operation
    return {"message": "Async operation completed"}
```

### Mixing Sync and Async

```python
@app.get("/sync-endpoint")
def sync_endpoint():
    """Synchronous endpoint for CPU-bound operations."""
    # CPU-intensive work here
    result = sum(range(1000000))
    return {"result": result}

@app.get("/async-with-sync")
async def async_with_sync():
    """Async endpoint that calls sync code."""
    # Use run_in_executor for CPU-bound operations
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, cpu_intensive_function)
    return {"result": result}

def cpu_intensive_function():
    """CPU-intensive function."""
    return sum(range(1000000))
```

## Database Async Patterns

### SQLAlchemy Async Session

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

# Create async engine
engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost/db",
    echo=True,
    pool_size=20,
    max_overflow=0,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Dependency for getting async session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Async database operations
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### Bulk Database Operations

```python
async def bulk_insert_users(db: AsyncSession, users_data: List[dict]):
    """Efficiently insert multiple users."""
    users = [User(**user_data) for user_data in users_data]
    db.add_all(users)
    await db.commit()
    return users

async def bulk_update_with_case(db: AsyncSession, updates: List[dict]):
    """Bulk update using CASE statement."""
    from sqlalchemy import case
    
    user_ids = [update["id"] for update in updates]
    status_values = [update["status"] for update in updates]
    
    stmt = (
        update(User)
        .where(User.id.in_(user_ids))
        .values(status=case(*[(User.id == update["id"], update["status"]) for update in updates]))
    )
    
    await db.execute(stmt)
    await db.commit()
```

## External API Integration

### Async HTTP Client

```python
import httpx
from fastapi import HTTPException

async def fetch_external_data(url: str) -> dict:
    """Fetch data from external API using async client."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail="Service unavailable")

# Usage in endpoint
@app.get("/external-data")
async def get_external_data():
    data = await fetch_external_data("https://api.example.com/data")
    return {"external_data": data}
```

### Concurrent API Calls

```python
async def fetch_multiple_apis_concurrently(urls: List[str]) -> List[dict]:
    """Fetch multiple APIs concurrently."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                results.append({"url": urls[i], "error": str(response)})
            elif response.status_code == 200:
                results.append({"url": urls[i], "data": response.json()})
            else:
                results.append({"url": urls[i], "error": f"HTTP {response.status_code}"})
        
        return results

@app.get("/concurrent-data")
async def get_concurrent_data():
    urls = [
        "https://api1.example.com/data",
        "https://api2.example.com/data",
        "https://api3.example.com/data"
    ]
    results = await fetch_multiple_apis_concurrently(urls)
    return {"results": results}
```

### Circuit Breaker Pattern

```python
from typing import Callable, Any
import time
import asyncio

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

# Usage
circuit_breaker = CircuitBreaker()

@app.get("/reliable-external-data")
async def get_reliable_external_data():
    try:
        data = await circuit_breaker.call(fetch_external_data, "https://api.example.com/data")
        return {"data": data}
    except Exception as e:
        # Fallback to cached data or default response
        return {"data": "fallback_data", "error": str(e)}
```

## Background Tasks

### Basic Background Tasks

```python
from fastapi import BackgroundTasks

async def send_email_async(email: str, subject: str, body: str):
    """Send email asynchronously."""
    # Simulate email sending
    await asyncio.sleep(2)
    print(f"Email sent to {email}: {subject}")

@app.post("/send-email")
async def send_email(
    email: str,
    subject: str,
    body: str,
    background_tasks: BackgroundTasks
):
    """Endpoint that schedules background email task."""
    background_tasks.add_task(send_email_async, email, subject, body)
    return {"message": "Email will be sent in background"}
```

### Advanced Background Tasks with Celery

```python
from celery import Celery
import redis

# Configure Celery
celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task
def process_data_task(data_id: int):
    """Process data in background using Celery."""
    # Long-running data processing
    import time
    time.sleep(10)
    return f"Data {data_id} processed successfully"

@app.post("/process-data/{data_id}")
async def process_data(data_id: int):
    """Schedule data processing task."""
    task = process_data_task.delay(data_id)
    return {"task_id": task.id, "message": "Processing started"}

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Check background task status."""
    task = process_data_task.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    }
```

## WebSocket Implementation

### Basic WebSocket Handler

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client {client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} left the chat")
```

### WebSocket with Authentication

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_websocket_user(websocket: WebSocket, token: str = None):
    """Authenticate WebSocket connection."""
    if token is None:
        # Try to get token from query parameters
        token = websocket.query_params.get("token")
    
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return None
    
    try:
        # Verify JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        return user_id
    except jwt.JWTError:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return None

@app.websocket("/ws/authenticated")
async def authenticated_websocket(
    websocket: WebSocket,
    token: str = None
):
    user_id = await get_websocket_user(websocket, token)
    if not user_id:
        return
    
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process authenticated user data
            await manager.send_personal_message(f"User {user_id}: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

## Performance Optimization

### Connection Pooling

```python
from sqlalchemy.pool import NullPool, StaticPool

# For production with connection pooling
engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost/db",
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# For testing (in-memory SQLite)
test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    poolclass=StaticPool,
    connect_args={"check_same_thread": False},
)
```

### Async Caching

```python
from functools import lru_cache
import aioredis
import json

class AsyncCache:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None
    
    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def disconnect(self):
        if self.redis:
            await self.redis.close()
    
    async def get(self, key: str) -> Any:
        if not self.redis:
            await self.connect()
        
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, expire: int = 3600):
        if not self.redis:
            await self.connect()
        
        await self.redis.set(key, json.dumps(value), ex=expire)
    
    async def delete(self, key: str):
        if not self.redis:
            await self.connect()
        
        await self.redis.delete(key)

# Usage with dependency injection
async def get_cache() -> AsyncCache:
    cache = AsyncCache("redis://localhost:6379/0")
    await cache.connect()
    try:
        yield cache
    finally:
        await cache.disconnect()

@app.get("/cached-data/{key}")
async def get_cached_data(key: str, cache: AsyncCache = Depends(get_cache)):
    cached_data = await cache.get(key)
    if cached_data:
        return {"cached": True, "data": cached_data}
    
    # Fetch fresh data
    fresh_data = await fetch_fresh_data(key)
    await cache.set(key, fresh_data)
    return {"cached": False, "data": fresh_data}
```

### Async Context Managers

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def database_transaction(db: AsyncSession):
    """Async context manager for database transactions."""
    try:
        yield db
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

# Usage
async def create_user_with_profile(user_data: dict, profile_data: dict):
    async with database_transaction(db) as session:
        user = User(**user_data)
        session.add(user)
        await session.flush()  # Get user ID
        
        profile = Profile(user_id=user.id, **profile_data)
        session.add(profile)
        
        return user
```

## Error Handling

### Async Exception Handling

```python
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

async def safe_async_operation(operation: Callable, *args, **kwargs):
    """Safely execute async operation with proper error handling."""
    try:
        return await operation(*args, **kwargs)
    except asyncio.TimeoutError:
        logger.error(f"Timeout in operation: {operation.__name__}")
        raise HTTPException(status_code=504, detail="Operation timed out")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error in operation {operation.__name__}: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in operation {operation.__name__}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Usage
@app.get("/safe-external-data")
async def get_safe_external_data():
    data = await safe_async_operation(fetch_external_data, "https://api.example.com/data")
    return {"data": data}
```

### Retry Logic with Exponential Backoff

```python
import random

async def retry_with_backoff(
    operation: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Any:
    """Retry async operation with exponential backoff."""
    
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Calculate delay with exponential backoff
            delay = min(
                base_delay * (exponential_base ** attempt),
                max_delay
            )
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random())
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)

# Usage
@app.get("/reliable-data")
async def get_reliable_data():
    data = await retry_with_backoff(lambda: fetch_external_data("https://api.example.com/data"))
    return {"data": data}
```

## Testing Async Code

### Async Test Fixtures

```python
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Create test database engine
test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def db_session():
    """Create async database session for testing."""
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()
```

### Testing Async Endpoints

```python
@pytest.mark.asyncio
async def test_async_endpoint(async_client: AsyncClient):
    """Test async endpoint."""
    response = await async_client.get("/async-endpoint")
    assert response.status_code == 200
    assert response.json()["message"] == "Async operation completed"

@pytest.mark.asyncio
async def test_concurrent_requests(async_client: AsyncClient):
    """Test multiple concurrent requests."""
    import asyncio
    
    # Create multiple concurrent requests
    tasks = [
        async_client.get("/async-endpoint"),
        async_client.get("/async-endpoint"),
        async_client.get("/async-endpoint")
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Verify all requests succeeded
    for response in responses:
        assert response.status_code == 200
```

### Testing WebSocket Connections

```python
@pytest.mark.asyncio
async def test_websocket_connection(async_client: AsyncClient):
    """Test WebSocket connection."""
    async with async_client.websocket_connect("/ws/test-client") as websocket:
        # Send message
        await websocket.send_text("Hello WebSocket")
        
        # Receive response
        data = await websocket.receive_text()
        assert "Hello WebSocket" in data
        
        # Close connection
        await websocket.close()
```

## Best Practices

1. **Use async for I/O-bound operations**: Database queries, external API calls, file operations
2. **Use sync for CPU-bound operations**: Complex calculations, data processing
3. **Proper connection management**: Always close database connections and HTTP clients
4. **Error handling**: Implement comprehensive error handling and logging
5. **Testing**: Write async tests for async code
6. **Performance monitoring**: Monitor async operation performance and resource usage
7. **Resource cleanup**: Use context managers and proper cleanup in finally blocks
8. **Circuit breakers**: Implement circuit breakers for external service calls
9. **Rate limiting**: Implement appropriate rate limiting for external APIs
10. **Caching**: Use async caching to improve performance

## Common Pitfalls to Avoid

1. **Mixing blocking and non-blocking code**: Don't call blocking operations from async functions
2. **Forgetting to await**: Always await async operations
3. **Not handling exceptions**: Implement proper exception handling
4. **Resource leaks**: Always close connections and clean up resources
5. **Overusing async**: Not everything needs to be async - use appropriately
6. **Ignoring timeouts**: Always set appropriate timeouts for external calls
7. **Not testing async code**: Write proper async tests
8. **Forgetting about thread safety**: Be aware of thread safety in async contexts