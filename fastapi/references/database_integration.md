# FastAPI Database Integration Reference

This document provides comprehensive guidance on integrating FastAPI applications with databases, covering SQLAlchemy setup, async patterns, connection management, and best practices.

## Table of Contents

1. [Database Configuration](#database-configuration)
2. [SQLAlchemy Integration](#sqlalchemy-integration)
3. [Async Database Operations](#async-database-operations)
4. [Connection Pooling](#connection-pooling)
5. [Database Models](#database-models)
6. [CRUD Operations](#crud-operations)
7. [Migrations](#migrations)
8. [Transaction Management](#transaction-management)
9. [Performance Optimization](#performance-optimization)
10. [Testing Database Integration](#testing-database-integration)

## Database Configuration

### Environment-Based Configuration

```python
from pydantic import BaseSettings
from typing import Optional

class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # Database connection
    DATABASE_URL: str = "postgresql://user:password@localhost/db"
    ASYNC_DATABASE_URL: Optional[str] = None
    
    # Connection pooling
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_POOL_PRE_PING: bool = True
    
    # Performance
    DB_ECHO: bool = False
    DB_ECHO_POOL: bool = False
    
    # Security
    DB_SECRET_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Generate async URL if not provided
        if not self.ASYNC_DATABASE_URL and self.DATABASE_URL:
            self.ASYNC_DATABASE_URL = self.DATABASE_URL.replace(
                "postgresql://", "postgresql+asyncpg://"
            )

# Usage
db_settings = DatabaseSettings()
```

### Multiple Database Support

```python
class MultiDatabaseSettings(BaseSettings):
    """Multi-database configuration."""
    
    # Primary database
    PRIMARY_DATABASE_URL: str
    PRIMARY_ASYNC_DATABASE_URL: Optional[str] = None
    
    # Replica database (read-only)
    REPLICA_DATABASE_URL: Optional[str] = None
    REPLICA_ASYNC_DATABASE_URL: Optional[str] = None
    
    # Analytics database
    ANALYTICS_DATABASE_URL: Optional[str] = None
    ANALYTICS_ASYNC_DATABASE_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Generate async URLs
        for prefix in ["PRIMARY", "REPLICA", "ANALYTICS"]:
            sync_url = getattr(self, f"{prefix}_DATABASE_URL", None)
            async_attr = f"{prefix}_ASYNC_DATABASE_URL"
            if sync_url and not getattr(self, async_attr, None):
                setattr(self, async_attr, sync_url.replace(
                    "postgresql://", "postgresql+asyncpg://"
                ))
```

## SQLAlchemy Integration

### Synchronous SQLAlchemy Setup

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Database settings
db_settings = DatabaseSettings()

# Create engine
engine = create_engine(
    db_settings.DATABASE_URL,
    pool_size=db_settings.DB_POOL_SIZE,
    max_overflow=db_settings.DB_MAX_OVERFLOW,
    pool_timeout=db_settings.DB_POOL_TIMEOUT,
    pool_recycle=db_settings.DB_POOL_RECYCLE,
    pool_pre_ping=db_settings.DB_POOL_PRE_PING,
    echo=db_settings.DB_ECHO,
    echo_pool=db_settings.DB_ECHO_POOL
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Dependency for database sessions
@contextmanager
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Usage in FastAPI
def get_user_service(db: Session = Depends(get_db)):
    """Get user service with database session."""
    return UserService(db)
```

### Async SQLAlchemy Setup

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import asynccontextmanager

# Create async engine
async_engine = create_async_engine(
    db_settings.ASYNC_DATABASE_URL,
    pool_size=db_settings.DB_POOL_SIZE,
    max_overflow=db_settings.DB_MAX_OVERFLOW,
    pool_timeout=db_settings.DB_POOL_TIMEOUT,
    pool_recycle=db_settings.DB_POOL_RECYCLE,
    pool_pre_ping=db_settings.DB_POOL_PRE_PING,
    echo=db_settings.DB_ECHO,
    echo_pool=db_settings.DB_ECHO_POOL
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

# Create base class for async models
AsyncBase = declarative_base()

# Dependency for async database sessions
@asynccontextmanager
async def get_async_db():
    """Get async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Usage in FastAPI async endpoints
async def get_user_service_async(db: AsyncSession = Depends(get_async_db)):
    """Get user service with async database session."""
    return AsyncUserService(db)
```

## Async Database Operations

### Basic Async CRUD Operations

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete
from typing import List, Optional

class AsyncUserRepository:
    """Async user repository."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, user_data: dict) -> User:
        """Create a new user."""
        stmt = insert(User).values(**user_data).returning(User)
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.scalar_one()
    
    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users with pagination."""
        stmt = select(User).offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def update(self, user_id: int, user_data: dict) -> Optional[User]:
        """Update user."""
        stmt = update(User).where(User.id == user_id).values(**user_data).returning(User)
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.scalar_one_or_none()
    
    async def delete(self, user_id: int) -> bool:
        """Delete user."""
        stmt = delete(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount > 0
```

### Advanced Async Queries

```python
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import joinedload, selectinload

class AsyncAdvancedUserRepository(AsyncUserRepository):
    """Advanced async user repository with complex queries."""
    
    async def get_active_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get active users."""
        stmt = select(User).where(User.is_active == True).offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_users_by_role(self, role: str) -> List[User]:
        """Get users by role."""
        stmt = select(User).join(User.roles).where(Role.name == role)
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def search_users(self, query: str) -> List[User]:
        """Search users by name or email."""
        stmt = select(User).where(
            or_(
                User.username.ilike(f"%{query}%"),
                User.email.ilike(f"%{query}%"),
                User.full_name.ilike(f"%{query}%")
            )
        )
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_user_with_relations(self, user_id: int) -> Optional[User]:
        """Get user with all relations."""
        stmt = select(User).options(
            selectinload(User.roles),
            selectinload(User.posts),
            selectinload(User.profile)
        ).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_user_statistics(self) -> dict:
        """Get user statistics."""
        stmt = select(
            func.count(User.id).label("total_users"),
            func.count(User.id).filter(User.is_active == True).label("active_users"),
            func.count(User.id).filter(User.created_at >= func.now() - timedelta(days=30)).label("new_users_30d")
        )
        result = await self.db.execute(stmt)
        row = result.one()
        return {
            "total_users": row.total_users,
            "active_users": row.active_users,
            "new_users_30d": row.new_users_30d
        }
```

## Connection Pooling

### Advanced Pool Configuration

```python
from sqlalchemy import pool
import asyncio

class AdvancedDatabasePool:
    """Advanced database connection pool configuration."""
    
    @staticmethod
    def create_optimized_engine(database_url: str, **kwargs) -> Engine:
        """Create optimized database engine with advanced pool settings."""
        
        # Pool configuration
        pool_config = {
            "pool_size": kwargs.get("pool_size", 20),
            "max_overflow": kwargs.get("max_overflow", 40),
            "pool_timeout": kwargs.get("pool_timeout", 30),
            "pool_recycle": kwargs.get("pool_recycle", 3600),
            "pool_pre_ping": kwargs.get("pool_pre_ping", True),
        }
        
        # Performance settings
        performance_config = {
            "echo": kwargs.get("echo", False),
            "echo_pool": kwargs.get("echo_pool", False),
            "logging_name": kwargs.get("logging_name", "sqlalchemy"),
        }
        
        # Connection settings
        connection_config = {
            "connect_args": kwargs.get("connect_args", {}),
            "execution_options": kwargs.get("execution_options", {}),
        }
        
        # Create engine with all configurations
        engine = create_engine(
            database_url,
            **pool_config,
            **performance_config,
            **connection_config
        )
        
        return engine
    
    @staticmethod
    def create_async_optimized_engine(database_url: str, **kwargs) -> AsyncEngine:
        """Create optimized async database engine."""
        
        # Pool configuration for async
        pool_config = {
            "pool_size": kwargs.get("pool_size", 20),
            "max_overflow": kwargs.get("max_overflow", 40),
            "pool_timeout": kwargs.get("pool_timeout", 30),
            "pool_recycle": kwargs.get("pool_recycle", 3600),
            "pool_pre_ping": kwargs.get("pool_pre_ping", True),
        }
        
        # Performance settings
        performance_config = {
            "echo": kwargs.get("echo", False),
            "echo_pool": kwargs.get("echo_pool", False),
        }
        
        # Create async engine
        async_engine = create_async_engine(
            database_url,
            **pool_config,
            **performance_config
        )
        
        return async_engine
```

### Connection Pool Monitoring

```python
import time
import logging
from sqlalchemy import event
from sqlalchemy.pool import Pool

logger = logging.getLogger(__name__)

class ConnectionPoolMonitor:
    """Monitor database connection pool performance."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.setup_event_listeners()
    
    def setup_event_listeners(self):
        """Setup SQLAlchemy event listeners."""
        
        @event.listens_for(Pool, "connect")
        def receive_connect(dbapi_connection, connection_record):
            logger.info(f"New connection created: {connection_record}")
        
        @event.listens_for(Pool, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            logger.debug(f"Connection checked out: {connection_record}")
        
        @event.listens_for(Pool, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            logger.debug(f"Connection checked in: {connection_record}")
        
        @event.listens_for(Pool, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            logger.warning(f"Connection invalidated: {connection_record}, Exception: {exception}")
    
    def get_pool_stats(self) -> dict:
        """Get current pool statistics."""
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow()
        }
```

## Database Models

### Base Model Classes

```python
from sqlalchemy import Column, Integer, DateTime, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Create base class
Base = declarative_base()

class BaseModel(Base):
    """Base model with common fields."""
    
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True, nullable=False)
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id})>"

class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    def soft_delete(self):
        """Mark record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self):
        """Restore deleted record."""
        self.is_deleted = False
        self.deleted_at = None

class TimestampMixin:
    """Mixin for timestamp fields."""
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

### User Model Example

```python
from sqlalchemy import Column, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

class User(BaseModel, SoftDeleteMixin):
    """User model."""
    
    __tablename__ = "users"
    
    # Basic information
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    bio = Column(Text, nullable=True)
    
    # Authentication
    hashed_password = Column(String(128), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    roles = relationship("Role", secondary="user_roles", back_populates="users")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

class UserProfile(BaseModel):
    """User profile model."""
    
    __tablename__ = "user_profiles"
    
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Profile information
    avatar_url = Column(String(500), nullable=True)
    website = Column(String(200), nullable=True)
    location = Column(String(100), nullable=True)
    birth_date = Column(DateTime(timezone=True), nullable=True)
    
    # Settings
    timezone = Column(String(50), default="UTC", nullable=False)
    language = Column(String(10), default="en", nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="profile")

class Role(BaseModel):
    """Role model."""
    
    __tablename__ = "roles"
    
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    
    # Relationships
    users = relationship("User", secondary="user_roles", back_populates="roles")
    permissions = relationship("Permission", secondary="role_permissions", back_populates="roles")

class UserRole(Base):
    """User-Role association table."""
    
    __tablename__ = "user_roles"
    
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)
    assigned_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
```

## CRUD Operations

### Generic CRUD Repository

```python
from typing import TypeVar, Generic, Type, Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete
from sqlalchemy.exc import IntegrityError

T = TypeVar("T", bound=Base)

class AsyncCRUDRepository(Generic[T]):
    """Generic async CRUD repository."""
    
    def __init__(self, model: Type[T], db: AsyncSession):
        self.model = model
        self.db = db
    
    async def create(self, obj_data: Dict[str, Any]) -> T:
        """Create a new record."""
        try:
            stmt = insert(self.model).values(**obj_data).returning(self.model)
            result = await self.db.execute(stmt)
            await self.db.commit()
            return result.scalar_one()
        except IntegrityError as e:
            await self.db.rollback()
            raise ValueError(f"Integrity error: {str(e)}")
    
    async def get_by_id(self, obj_id: Any) -> Optional[T]:
        """Get record by ID."""
        stmt = select(self.model).where(self.model.id == obj_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> List[T]:
        """Get all records with optional filters."""
        stmt = select(self.model)
        
        # Apply filters
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    stmt = stmt.where(getattr(self.model, field) == value)
        
        # Apply ordering
        if order_by:
            if hasattr(self.model, order_by):
                stmt = stmt.order_by(getattr(self.model, order_by))
        
        # Apply pagination
        stmt = stmt.offset(skip).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def update(self, obj_id: Any, obj_data: Dict[str, Any]) -> Optional[T]:
        """Update record by ID."""
        try:
            stmt = update(self.model).where(self.model.id == obj_id).values(**obj_data).returning(self.model)
            result = await self.db.execute(stmt)
            await self.db.commit()
            return result.scalar_one_or_none()
        except IntegrityError as e:
            await self.db.rollback()
            raise ValueError(f"Integrity error: {str(e)}")
    
    async def delete(self, obj_id: Any) -> bool:
        """Delete record by ID."""
        stmt = delete(self.model).where(self.model.id == obj_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount > 0
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filters."""
        stmt = select(func.count(self.model.id))
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    stmt = stmt.where(getattr(self.model, field) == value)
        
        result = await self.db.execute(stmt)
        return result.scalar()
```

### Service Layer Pattern

```python
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class UserCreate(BaseModel):
    """User creation model."""
    username: str
    email: str
    full_name: Optional[str] = None
    password: str

class UserUpdate(BaseModel):
    """User update model."""
    username: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None

class UserService:
    """User service layer."""
    
    def __init__(self, repository: AsyncCRUDRepository[User]):
        self.repository = repository
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        # Hash password
        hashed_password = self._hash_password(user_data.password)
        
        # Create user data
        user_dict = user_data.dict()
        user_dict["hashed_password"] = hashed_password
        del user_dict["password"]
        
        return await self.repository.create(user_dict)
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return await self.repository.get_by_id(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        users = await self.repository.get_all(filters={"email": email})
        return users[0] if users else None
    
    async def update_user(self, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """Update user."""
        update_dict = {k: v for k, v in user_data.dict().items() if v is not None}
        return await self.repository.update(user_id, update_dict)
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete user."""
        return await self.repository.delete(user_id)
    
    async def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users."""
        return await self.repository.get_all(skip=skip, limit=limit)
    
    def _hash_password(self, password: str) -> str:
        """Hash password."""
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)
```

## Migrations

### Alembic Configuration

```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import your models and database configuration
from database.models import Base
from database.config import DATABASE_URL

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = DATABASE_URL
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Migration Script Generation

```python
# alembic/script.py.mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

### Migration Commands

```bash
# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Add user table"

# Apply migrations
alembic upgrade head

# Downgrade migration
alembic downgrade -1

# Show migration history
alembic history

# Show current migration
alembic current
```

## Transaction Management

### Manual Transaction Management

```python
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager

class TransactionManager:
    """Transaction manager for async database operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self.db
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            raise e
    
    async def execute_in_transaction(self, func, *args, **kwargs):
        """Execute function within a transaction."""
        async with self.transaction():
            return await func(*args, **kwargs)

# Usage
async def create_user_with_profile(user_data: dict, profile_data: dict):
    """Create user with profile in a transaction."""
    async with get_async_db() as db:
        tx_manager = TransactionManager(db)
        
        async with tx_manager.transaction():
            # Create user
            user = User(**user_data)
            db.add(user)
            await db.flush()  # Get user ID
            
            # Create profile
            profile = UserProfile(user_id=user.id, **profile_data)
            db.add(profile)
            
            return user
```

### Automatic Transaction Management

```python
from functools import wraps
from typing import Callable, Any

def transactional(func: Callable) -> Callable:
    """Decorator for automatic transaction management."""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get database session from arguments
        db = None
        for arg in args:
            if isinstance(arg, AsyncSession):
                db = arg
                break
        
        if db is None:
            raise ValueError("No AsyncSession found in arguments")
        
        try:
            result = await func(*args, **kwargs)
            await db.commit()
            return result
        except Exception as e:
            await db.rollback()
            raise e
    
    return wrapper

# Usage
@transactional
async def create_order(db: AsyncSession, order_data: dict, items_data: List[dict]):
    """Create order with automatic transaction management."""
    # Create order
    order = Order(**order_data)
    db.add(order)
    await db.flush()
    
    # Create order items
    for item_data in items_data:
        item_data["order_id"] = order.id
        order_item = OrderItem(**item_data)
        db.add(order_item)
    
    return order
```

## Performance Optimization

### Query Optimization

```python
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload, selectinload, subqueryload

class OptimizedUserRepository:
    """Optimized user repository with query performance improvements."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_user_with_eager_loading(self, user_id: int) -> Optional[User]:
        """Get user with eager loading of relationships."""
        stmt = select(User).options(
            selectinload(User.roles),
            selectinload(User.posts),
            selectinload(User.profile)
        ).where(User.id == user_id)
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_users_with_post_count(self, skip: int = 0, limit: int = 100) -> List[dict]:
        """Get users with post count using subquery."""
        # Create subquery for post count
        post_count_subquery = (
            select(
                Post.author_id,
                func.count(Post.id).label("post_count")
            )
            .group_by(Post.author_id)
            .subquery()
        )
        
        # Main query with join to subquery
        stmt = (
            select(
                User.id,
                User.username,
                User.email,
                post_count_subquery.c.post_count
            )
            .outerjoin(post_count_subquery, User.id == post_count_subquery.c.author_id)
            .offset(skip)
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        return [
            {
                "id": row.id,
                "username": row.username,
                "email": row.email,
                "post_count": row.post_count or 0
            }
            for row in result.all()
        ]
    
    async def bulk_create_users(self, users_data: List[dict]) -> List[User]:
        """Bulk create users for better performance."""
        stmt = insert(User).returning(User)
        result = await self.db.execute(stmt, users_data)
        await self.db.commit()
        return result.scalars().all()
```

### Caching Integration

```python
import aioredis
import json
from typing import Optional, Any

class CachedUserRepository:
    """User repository with caching support."""
    
    def __init__(self, db: AsyncSession, redis: aioredis.Redis):
        self.db = db
        self.redis = redis
        self.cache_ttl = 3600  # 1 hour
    
    async def get_user_cached(self, user_id: int) -> Optional[User]:
        """Get user with caching."""
        cache_key = f"user:{user_id}"
        
        # Try to get from cache
        cached_data = await self.redis.get(cache_key)
        if cached_data:
            user_dict = json.loads(cached_data)
            return User(**user_dict)
        
        # Get from database
        user = await self.get_user(user_id)
        if user:
            # Cache the user data
            user_dict = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active
            }
            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(user_dict)
            )
        
        return user
    
    async def invalidate_user_cache(self, user_id: int):
        """Invalidate user cache."""
        cache_key = f"user:{user_id}"
        await self.redis.delete(cache_key)
    
    async def update_user_with_cache(self, user_id: int, user_data: dict) -> Optional[User]:
        """Update user and invalidate cache."""
        user = await self.update_user(user_id, user_data)
        if user:
            await self.invalidate_user_cache(user_id)
        return user
```

## Testing Database Integration

### Test Database Setup

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import tempfile
import os

@pytest.fixture(scope="session")
def test_db_path():
    """Create temporary test database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        yield tmp.name
    os.unlink(tmp.name)

@pytest.fixture(scope="session")
def test_engine(test_db_path):
    """Create test database engine."""
    engine = create_engine(f"sqlite:///{test_db_path}")
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()

@pytest.fixture(scope="session")
def async_test_engine(test_db_path):
    """Create async test database engine."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{test_db_path}")
    yield engine
    engine.dispose()

@pytest.fixture
def db_session(test_engine):
    """Create database session for testing."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
async def async_db_session(async_test_engine):
    """Create async database session for testing."""
    AsyncSessionLocal = sessionmaker(
        bind=async_test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    async with AsyncSessionLocal() as session:
        yield session
```

### Test CRUD Operations

```python
import pytest
from sqlalchemy import select

class TestUserRepository:
    """Test cases for user repository."""
    
    @pytest.fixture
    def user_repository(self, async_db_session):
        """Create user repository for testing."""
        return AsyncUserRepository(async_db_session)
    
    @pytest.mark.asyncio
    async def test_create_user(self, user_repository):
        """Test user creation."""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "full_name": "Test User",
            "hashed_password": "hashed_password"
        }
        
        user = await user_repository.create(user_data)
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_get_user_by_id(self, user_repository):
        """Test getting user by ID."""
        # Create a user first
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed_password"
        }
        created_user = await user_repository.create(user_data)
        
        # Get user by ID
        user = await user_repository.get_by_id(created_user.id)
        
        assert user is not None
        assert user.id == created_user.id
        assert user.username == "testuser"
    
    @pytest.mark.asyncio
    async def test_update_user(self, user_repository):
        """Test user update."""
        # Create a user first
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed_password"
        }
        user = await user_repository.create(user_data)
        
        # Update user
        update_data = {"full_name": "Updated Name"}
        updated_user = await user_repository.update(user.id, update_data)
        
        assert updated_user is not None
        assert updated_user.full_name == "Updated Name"
    
    @pytest.mark.asyncio
    async def test_delete_user(self, user_repository):
        """Test user deletion."""
        # Create a user first
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed_password"
        }
        user = await user_repository.create(user_data)
        
        # Delete user
        deleted = await user_repository.delete(user.id)
        
        assert deleted is True
        
        # Verify user is deleted
        deleted_user = await user_repository.get_by_id(user.id)
        assert deleted_user is None
    
    @pytest.mark.asyncio
    async def test_get_all_users_with_pagination(self, user_repository):
        """Test getting all users with pagination."""
        # Create multiple users
        for i in range(5):
            user_data = {
                "username": f"testuser{i}",
                "email": f"test{i}@example.com",
                "hashed_password": "hashed_password"
            }
            await user_repository.create(user_data)
        
        # Get users with pagination
        users_page_1 = await user_repository.get_all(skip=0, limit=2)
        users_page_2 = await user_repository.get_all(skip=2, limit=2)
        
        assert len(users_page_1) == 2
        assert len(users_page_2) == 2
        assert users_page_1[0].username == "testuser0"
        assert users_page_2[0].username == "testuser2"
```

### Test Database Transactions

```python
import pytest
from sqlalchemy.exc import IntegrityError

class TestTransactionManagement:
    """Test database transaction management."""
    
    @pytest.fixture
    def transaction_manager(self, async_db_session):
        """Create transaction manager for testing."""
        return TransactionManager(async_db_session)
    
    @pytest.mark.asyncio
    async def test_successful_transaction(self, transaction_manager):
        """Test successful transaction."""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed_password"
        }
        
        async def create_user():
            return await transaction_manager.db.execute(
                insert(User).values(**user_data).returning(User)
            )
        
        result = await transaction_manager.execute_in_transaction(create_user)
        user = result.scalar_one()
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_failed_transaction_rollback(self, transaction_manager):
        """Test failed transaction rollback."""
        # Create a user first
        user_data = {
            "username": "existinguser",
            "email": "existing@example.com",
            "hashed_password": "hashed_password"
        }
        
        await transaction_manager.db.execute(insert(User).values(**user_data))
        await transaction_manager.db.commit()
        
        # Try to create another user with the same email (should fail)
        duplicate_data = {
            "username": "newuser",
            "email": "existing@example.com",  # Duplicate email
            "hashed_password": "hashed_password"
        }
        
        async def create_duplicate_user():
            return await transaction_manager.db.execute(
                insert(User).values(**duplicate_data)
            )
        
        with pytest.raises(IntegrityError):
            await transaction_manager.execute_in_transaction(create_duplicate_user)
        
        # Verify no duplicate user was created
        result = await transaction_manager.db.execute(
            select(User).where(User.email == "existing@example.com")
        )
        users = result.scalars().all()
        assert len(users) == 1  # Only the original user exists
```

## Best Practices

1. **Use connection pooling**: Configure appropriate pool sizes for your application load
2. **Implement proper error handling**: Handle database errors gracefully with rollback
3. **Use async operations**: Leverage async database operations for better performance
4. **Implement caching**: Cache frequently accessed data to reduce database load
5. **Monitor performance**: Track query performance and optimize slow queries
6. **Use transactions**: Wrap related operations in transactions for data consistency
7. **Implement soft deletes**: Use soft deletes for data integrity and audit trails
8. **Version your database**: Use migrations to manage database schema changes
9. **Test thoroughly**: Write comprehensive tests for database operations
10. **Secure your database**: Use proper authentication and connection encryption