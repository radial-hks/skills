#!/usr/bin/env python3
"""
FastAPI Database Setup Script

This script provides comprehensive database configurations for FastAPI applications,
including SQLAlchemy setup, database models, connection pooling, migrations, and
async database operations.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

class FastAPIDatabaseSetup:
    """Generate database configurations for FastAPI applications."""
    
    def __init__(self):
        """Initialize database setup."""
        self.database_url = "sqlite:///./app.db"  # Default SQLite
        self.async_database_url = "sqlite+aiosqlite:///./app.db"
    
    def generate_sqlalchemy_config(self, project_name: str) -> str:
        """Generate SQLAlchemy database configuration."""
        
        code = f'''"""
SQLAlchemy Database Configuration for {project_name}

This module provides SQLAlchemy database setup with connection pooling,
session management, and async support.
"""

from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator, Optional
import logging

# Database configuration
DATABASE_URL = "{self.database_url}"
ASYNC_DATABASE_URL = "{self.async_database_url}"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

# Engine configuration
engine_config = {{
    "echo": True,  # Set to False in production
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "pool_size": 10,
    "max_overflow": 20,
}}

# Handle SQLite-specific settings
if DATABASE_URL.startswith("sqlite"):
    if DATABASE_URL.startswith("sqlite:///./"):
        # Create directory for SQLite file if it doesn't exist
        import os
        db_path = DATABASE_URL.replace("sqlite:///./", "")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    engine_config.update({{
        "connect_args": {{"check_same_thread": False}},
        "poolclass": StaticPool,
    }})

# Create engine
engine = create_engine(DATABASE_URL, **engine_config)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {{e}}")
        raise

def check_db_connection() -> bool:
    """Check database connection."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {{e}}")
        return False

def get_table_names() -> list:
    """Get all table names in the database."""
    inspector = inspect(engine)
    return inspector.get_table_names()

def get_table_info(table_name: str) -> dict:
    """Get information about a specific table."""
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return {{
        "name": table_name,
        "columns": [{{"name": col["name"], "type": str(col["type"])}} for col in columns]
    }}

# Import inspect here to avoid circular imports
from sqlalchemy import inspect

# Database utilities
class DatabaseManager:
    """Database management utilities."""
    
    def __init__(self, engine=engine):
        self.engine = engine
    
    def create_backup(self, backup_path: str) -> bool:
        """Create database backup."""
        try:
            import shutil
            if self.engine.url.database:
                shutil.copy2(self.engine.url.database, backup_path)
                logger.info(f"Database backup created: {{backup_path}}")
                return True
            return False
        except Exception as e:
            logger.error(f"Backup creation failed: {{e}}")
            return False
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore database from backup."""
        try:
            import shutil
            if self.engine.url.database:
                shutil.copy2(backup_path, self.engine.url.database)
                logger.info(f"Database restored from: {{backup_path}}")
                return True
            return False
        except Exception as e:
            logger.error(f"Database restore failed: {{e}}")
            return False
    
    def get_database_size(self) -> Optional[int]:
        """Get database size in bytes."""
        try:
            if self.engine.url.database and os.path.exists(self.engine.url.database):
                return os.path.getsize(self.engine.url.database)
            return None
        except Exception as e:
            logger.error(f"Failed to get database size: {{e}}")
            return None
    
    def vacuum_database(self) -> bool:
        """Vacuum SQLite database to reclaim space."""
        try:
            if self.engine.url.drivername == "sqlite":
                with self.engine.connect() as connection:
                    connection.execute(text("VACUUM"))
                logger.info("Database vacuumed successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Database vacuum failed: {{e}}")
            return False

# Global database manager instance
db_manager = DatabaseManager()
'''
        return code
    
    def generate_async_sqlalchemy_config(self) -> str:
        """Generate async SQLAlchemy database configuration."""
        
        code = '''"""
Async SQLAlchemy Database Configuration

This module provides async SQLAlchemy database setup for FastAPI applications.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from typing import AsyncGenerator
import logging

# Async database configuration
ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./app.db"

# Logging setup
logger = logging.getLogger(__name__)

# Create declarative base for async
AsyncBase = declarative_base()

# Async engine configuration
async_engine_config = {
    "echo": True,  # Set to False in production
    "pool_pre_ping": True,
    "pool_recycle": 3600,
}

# Create async engine
async_engine = create_async_engine(ASYNC_DATABASE_URL, **async_engine_config)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_async_db():
    """Initialize async database tables."""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(AsyncBase.metadata.create_all)
        logger.info("Async database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating async database tables: {e}")
        raise

async def check_async_db_connection() -> bool:
    """Check async database connection."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Async database connection successful")
        return True
    except Exception as e:
        logger.error(f"Async database connection failed: {e}")
        return False

# Async database utilities
class AsyncDatabaseManager:
    """Async database management utilities."""
    
    def __init__(self, engine=async_engine):
        self.engine = engine
    
    async def execute_query(self, query: str, params: dict = None) -> list:
        """Execute raw SQL query."""
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text(query), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def get_table_count(self, table_name: str) -> int:
        """Get row count for a table."""
        try:
            result = await self.execute_query(
                f"SELECT COUNT(*) as count FROM {{table_name}}"
            )
            return result[0]["count"] if result else 0
        except Exception as e:
            logger.error(f"Failed to get table count: {e}")
            return 0
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        try:
            result = await self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name",
                {{"table_name": table_name}}
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Failed to check table existence: {e}")
            return False

# Global async database manager instance
async_db_manager = AsyncDatabaseManager()

# Import text here to avoid circular imports
from sqlalchemy import text
'''
        return code
    
    def generate_database_models_template(self) -> str:
        """Generate database models template."""
        
        code = '''"""
Database Models Template

This module provides example database models for FastAPI applications.
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from .database import Base

# Many-to-Many relationship table
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)

class User(Base):
    """User model."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    items = relationship("Item", back_populates="owner")
    
    def __repr__(self):
        return f"<User(username='{{self.username}}', email='{{self.email}}')>"

class Role(Base):
    """Role model."""
    
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, index=True, nullable=False)
    description = Column(String(200), nullable=True)
    permissions = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    
    def __repr__(self):
        return f"<Role(name='{{self.name}}')>"

class Item(Base):
    """Item model."""
    
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), index=True, nullable=False)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=False)
    is_available = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    owner = relationship("User", back_populates="items")
    
    def __repr__(self):
        return f"<Item(title='{{self.title}}', price={{self.price}})>"

class AuditLog(Base):
    """Audit log model."""
    
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(100), nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<AuditLog(action='{{self.action}}', user_id={{self.user_id}})>"

# Async models (if using async SQLAlchemy)
class AsyncUser:
    """Async user model mixin."""
    
    @classmethod
    async def get_by_username(cls, db_session, username: str):
        """Get user by username."""
        from sqlalchemy import select
        result = await db_session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()
    
    @classmethod
    async def get_by_email(cls, db_session, email: str):
        """Get user by email."""
        from sqlalchemy import select
        result = await db_session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    @classmethod
    async def create(cls, db_session, **kwargs):
        """Create new user."""
        user = User(**kwargs)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        return user

# Database indexes and constraints
from sqlalchemy import Index

# Create indexes for better performance
Index("idx_user_email_active", User.email, User.is_active)
Index("idx_item_price_available", Item.price, Item.is_available)
Index("idx_audit_log_user_action", AuditLog.user_id, AuditLog.action)
Index("idx_audit_log_created", AuditLog.created_at)
'''
        return code
    
    def generate_database_crud_operations(self) -> str:
        """Generate database CRUD operations."""
        
        code = '''"""
Database CRUD Operations

This module provides generic CRUD operations for database models.
"""

from typing import TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType")

class CRUDBase:
    """Base CRUD operations."""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """Get record by ID."""
        return db.query(self.model).filter(self.model.id == id).first()
    
    def get_multi(
        self, 
        db: Session, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Dict[str, Any] = None,
        order_by: str = None,
        order_desc: bool = False
    ) -> List[ModelType]:
        """Get multiple records with pagination and filtering."""
        query = db.query(self.model)
        
        # Apply filters
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key) and value is not None:
                    filter_conditions.append(getattr(self.model, key) == value)
            
            if filter_conditions:
                query = query.filter(and_(*filter_conditions))
        
        # Apply ordering
        if order_by and hasattr(self.model, order_by):
            order_field = getattr(self.model, order_by)
            if order_desc:
                query = query.order_by(desc(order_field))
            else:
                query = query.order_by(asc(order_field))
        
        return query.offset(skip).limit(limit).all()
    
    def create(self, db: Session, *, obj_in: Dict[str, Any]) -> ModelType:
        """Create new record."""
        try:
            db_obj = self.model(**obj_in)
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Record already exists or violates constraints"
            )
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    def update(
        self, 
        db: Session, 
        *, 
        db_obj: ModelType, 
        obj_in: Dict[str, Any]
    ) -> ModelType:
        """Update existing record."""
        try:
            # Update fields
            for field, value in obj_in.items():
                if hasattr(db_obj, field) and value is not None:
                    setattr(db_obj, field, value)
            
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error updating {self.model.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Update violates constraints"
            )
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating {self.model.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    def delete(self, db: Session, *, id: Any) -> bool:
        """Delete record by ID."""
        try:
            obj = db.query(self.model).filter(self.model.id == id).first()
            if not obj:
                return False
            
            db.delete(obj)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting {self.model.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    def count(self, db: Session, filters: Dict[str, Any] = None) -> int:
        """Count records with optional filters."""
        query = db.query(self.model)
        
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key) and value is not None:
                    filter_conditions.append(getattr(self.model, key) == value)
            
            if filter_conditions:
                query = query.filter(and_(*filter_conditions))
        
        return query.count()
    
    def exists(self, db: Session, filters: Dict[str, Any]) -> bool:
        """Check if records exist with given filters."""
        return self.count(db, filters) > 0

# Specific CRUD operations for models
from .models import User, Item, Role, AuditLog

class CRUDUser(CRUDBase):
    """CRUD operations for User model."""
    
    def __init__(self):
        super().__init__(User)
    
    def get_by_username(self, db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()
    
    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()
    
    def get_active_users(self, db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """Get active users."""
        return db.query(User).filter(User.is_active == True).offset(skip).limit(limit).all()
    
    def create_with_roles(self, db: Session, *, obj_in: Dict[str, Any], role_ids: List[int]) -> User:
        """Create user with roles."""
        user = self.create(db, obj_in=obj_in)
        
        # Add roles
        if role_ids:
            from .models import Role
            roles = db.query(Role).filter(Role.id.in_(role_ids)).all()
            user.roles.extend(roles)
            db.commit()
            db.refresh(user)
        
        return user

class CRUDItem(CRUDBase):
    """CRUD operations for Item model."""
    
    def __init__(self):
        super().__init__(Item)
    
    def get_available_items(self, db: Session, skip: int = 0, limit: int = 100) -> List[Item]:
        """Get available items."""
        return db.query(Item).filter(Item.is_available == True).offset(skip).limit(limit).all()
    
    def get_by_owner(self, db: Session, owner_id: int, skip: int = 0, limit: int = 100) -> List[Item]:
        """Get items by owner."""
        return db.query(Item).filter(Item.owner_id == owner_id).offset(skip).limit(limit).all()
    
    def search_items(self, db: Session, query: str, skip: int = 0, limit: int = 100) -> List[Item]:
        """Search items by title or description."""
        return db.query(Item).filter(
            or_(
                Item.title.contains(query),
                Item.description.contains(query)
            )
        ).offset(skip).limit(limit).all()

class CRUDRole(CRUDBase):
    """CRUD operations for Role model."""
    
    def __init__(self):
        super().__init__(Role)
    
    def get_by_name(self, db: Session, name: str) -> Optional[Role]:
        """Get role by name."""
        return db.query(Role).filter(Role.name == name).first()

class CRUDAuditLog(CRUDBase):
    """CRUD operations for AuditLog model."""
    
    def __init__(self):
        super().__init__(AuditLog)
    
    def get_by_user(self, db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        """Get audit logs by user."""
        return db.query(AuditLog).filter(AuditLog.user_id == user_id).order_by(
            desc(AuditLog.created_at)
        ).offset(skip).limit(limit).all()
    
    def get_by_action(self, db: Session, action: str, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        """Get audit logs by action."""
        return db.query(AuditLog).filter(AuditLog.action == action).order_by(
            desc(AuditLog.created_at)
        ).offset(skip).limit(limit).all()
    
    def create_audit_log(self, db: Session, *, user_id: int, action: str, **kwargs) -> AuditLog:
        """Create audit log entry."""
        log_data = {
            "user_id": user_id,
            "action": action,
            **kwargs
        }
        return self.create(db, obj_in=log_data)

# Global CRUD instances
crud_user = CRUDUser()
crud_item = CRUDItem()
crud_role = CRUDRole()
crud_audit_log = CRUDAuditLog()

# Async CRUD operations (for async database)
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

class AsyncCRUDBase:
    """Base async CRUD operations."""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def get(self, db: AsyncSession, id: Any) -> Optional[ModelType]:
        """Get record by ID."""
        from sqlalchemy import select
        result = await db.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()
    
    async def get_multi(
        self, 
        db: AsyncSession, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Dict[str, Any] = None
    ) -> List[ModelType]:
        """Get multiple records with pagination and filtering."""
        from sqlalchemy import select, and_
        
        query = select(self.model)
        
        # Apply filters
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key) and value is not None:
                    filter_conditions.append(getattr(self.model, key) == value)
            
            if filter_conditions:
                query = query.where(and_(*filter_conditions))
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    
    async def create(self, db: AsyncSession, *, obj_in: Dict[str, Any]) -> ModelType:
        """Create new record."""
        try:
            db_obj = self.model(**obj_in)
            db.add(db_obj)
            await db.commit()
            await db.refresh(db_obj)
            return db_obj
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )

# Async CRUD instances
async_crud_user = AsyncCRUDBase(User)
async_crud_item = AsyncCRUDBase(Item)
'''
        return code
    
    def generate_migrations_template(self) -> str:
        """Generate database migrations template."""
        
        code = '''"""
Database Migrations Template

This module provides database migration utilities for FastAPI applications.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MigrationManager:
    """Simple migration manager for SQLite databases."""
    
    def __init__(self, db_path: str, migrations_dir: str = "migrations"):
        self.db_path = db_path
        self.migrations_dir = migrations_dir
        self.applied_migrations_table = "applied_migrations"
        
        # Create migrations directory
        os.makedirs(migrations_dir, exist_ok=True)
    
    def create_migration(self, name: str, up_sql: str, down_sql: str = "") -> str:
        """Create a new migration file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{{timestamp}}_{{name}}.json"
        filepath = os.path.join(self.migrations_dir, filename)
        
        migration_data = {
            "name": name,
            "timestamp": timestamp,
            "up_sql": up_sql,
            "down_sql": down_sql,
            "created_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(migration_data, f, indent=2)
        
        logger.info(f"Migration created: {{filepath}}")
        return filepath
    
    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create migrations table if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {{self.applied_migrations_table}} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
            """)
            
            # Get applied migrations
            cursor.execute(f"SELECT name FROM {{self.applied_migrations_table}}")
            applied = {row[0] for row in cursor.fetchall()}
            
            # Get all migration files
            migration_files = []
            for filename in os.listdir(self.migrations_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.migrations_dir, filename)
                    with open(filepath, 'r') as f:
                        migration = json.load(f)
                        migration_files.append({
                            'name': migration['name'],
                            'filename': filename,
                            'filepath': filepath,
                            'timestamp': migration['timestamp']
                        })
            
            # Sort by timestamp
            migration_files.sort(key=lambda x: x['timestamp'])
            
            # Get pending migrations
            pending = [
                file_info for file_info in migration_files
                if file_info['name'] not in applied
            ]
            
            conn.close()
            return pending
            
        except Exception as e:
            logger.error(f"Error getting pending migrations: {{e}}")
            return []
    
    def apply_migration(self, migration_file: str) -> bool:
        """Apply a migration."""
        try:
            import sqlite3
            
            with open(migration_file, 'r') as f:
                migration = json.load(f)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute migration SQL
            if migration['up_sql']:
                cursor.executescript(migration['up_sql'])
            
            # Record applied migration
            cursor.execute(f"""
                INSERT INTO {{self.applied_migrations_table}} (name, timestamp, applied_at)
                VALUES (?, ?, ?)
            """, (migration['name'], migration['timestamp'], datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Migration applied: {{migration['name']}}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying migration {{migration_file}}: {{e}}")
            return False
    
    def apply_all_migrations(self) -> int:
        """Apply all pending migrations."""
        pending = self.get_pending_migrations()
        applied_count = 0
        
        for migration in pending:
            if self.apply_migration(migration['filepath']):
                applied_count += 1
            else:
                logger.error(f"Failed to apply migration: {{migration['name']}}")
                break
        
        logger.info(f"Applied {{applied_count}} migrations")
        return applied_count
    
    def rollback_migration(self, migration_name: str) -> bool:
        """Rollback a specific migration."""
        try:
            import sqlite3
            
            # Find migration file
            migration_file = None
            for filename in os.listdir(self.migrations_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.migrations_dir, filename)
                    with open(filepath, 'r') as f:
                        migration = json.load(f)
                        if migration['name'] == migration_name:
                            migration_file = filepath
                            break
            
            if not migration_file:
                logger.error(f"Migration not found: {{migration_name}}")
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute rollback SQL
            with open(migration_file, 'r') as f:
                migration = json.load(f)
                if migration['down_sql']:
                    cursor.executescript(migration['down_sql'])
            
            # Remove applied migration record
            cursor.execute(f"""
                DELETE FROM {{self.applied_migrations_table}}
                WHERE name = ?
            """, (migration_name,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Migration rolled back: {{migration_name}}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back migration {{migration_name}}: {{e}}")
            return False

# Example migration creation functions
def create_user_table_migration() -> str:
    """Create migration for users table."""
    up_sql = """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        full_name VARCHAR(100),
        hashed_password VARCHAR(128) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        is_superuser BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX idx_users_username ON users(username);
    CREATE INDEX idx_users_email ON users(email);
    CREATE INDEX idx_users_active ON users(is_active);
    """
    
    down_sql = """
    DROP INDEX IF EXISTS idx_users_active;
    DROP INDEX IF EXISTS idx_users_email;
    DROP INDEX IF EXISTS idx_users_username;
    DROP TABLE IF EXISTS users;
    """
    
    return up_sql, down_sql

def create_items_table_migration() -> str:
    """Create migration for items table."""
    up_sql = """
    CREATE TABLE items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title VARCHAR(100) NOT NULL,
        description TEXT,
        price REAL NOT NULL,
        is_available BOOLEAN DEFAULT TRUE,
        owner_id INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (owner_id) REFERENCES users (id)
    );
    
    CREATE INDEX idx_items_owner ON items(owner_id);
    CREATE INDEX idx_items_available ON items(is_available);
    CREATE INDEX idx_items_price ON items(price);
    """
    
    down_sql = """
    DROP INDEX IF EXISTS idx_items_price;
    DROP INDEX IF EXISTS idx_items_available;
    DROP INDEX IF EXISTS idx_items_owner;
    DROP TABLE IF EXISTS items;
    """
    
    return up_sql, down_sql

# Migration CLI interface
def main():
    """CLI interface for migration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration manager")
    parser.add_argument("--db-path", default="app.db", help="Database file path")
    parser.add_argument("--migrations-dir", default="migrations", help="Migrations directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create migration command
    create_parser = subparsers.add_parser("create", help="Create a new migration")
    create_parser.add_argument("name", help="Migration name")
    create_parser.add_argument("--up-sql", help="Up SQL script")
    create_parser.add_argument("--down-sql", help="Down SQL script")
    
    # Apply migrations command
    apply_parser = subparsers.add_parser("apply", help="Apply migrations")
    apply_parser.add_argument("--all", action="store_true", help="Apply all pending migrations")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show migration status")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback a migration")
    rollback_parser.add_argument("name", help="Migration name to rollback")
    
    args = parser.parse_args()
    
    if args.command == "create":
        manager = MigrationManager(args.db_path, args.migrations_dir)
        up_sql = args.up_sql or ""
        down_sql = args.down_sql or ""
        migration_file = manager.create_migration(args.name, up_sql, down_sql)
        print(f"Migration created: {{migration_file}}")
        
    elif args.command == "apply":
        manager = MigrationManager(args.db_path, args.migrations_dir)
        if args.all:
            applied = manager.apply_all_migrations()
            print(f"Applied {{applied}} migrations")
        else:
            pending = manager.get_pending_migrations()
            if pending:
                print(f"{{len(pending)}} pending migrations:")
                for migration in pending:
                    print(f"  - {{migration['name']}}")
            else:
                print("No pending migrations")
    
    elif args.command == "status":
        manager = MigrationManager(args.db_path, args.migrations_dir)
        pending = manager.get_pending_migrations()
        if pending:
            print(f"{{len(pending)}} pending migrations:")
            for migration in pending:
                print(f"  - {{migration['name']}}")
        else:
            print("No pending migrations")
    
    elif args.command == "rollback":
        manager = MigrationManager(args.db_path, args.migrations_dir)
        success = manager.rollback_migration(args.name)
        if success:
            print(f"Migration rolled back: {{args.name}}")
        else:
            print(f"Failed to rollback migration: {{args.name}}")

if __name__ == "__main__":
    main()
'''
        return code
    
    def generate_database_config_file(self) -> str:
        """Generate main database configuration file."""
        
        code = f'''"""
Database Configuration for FastAPI Application

This module provides comprehensive database setup for FastAPI applications.
"""

from typing import Optional, Generator, AsyncGenerator
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

# Database URLs (configure for your environment)
DATABASE_URL = "{self.database_url}"
ASYNC_DATABASE_URL = "{self.async_database_url}"

# Database connection settings
DB_CONFIG = {{
    "pool_size": 10,
    "max_overflow": 20,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "echo": False  # Set to True for SQL logging
}}

# Import database components
from .database.core import get_db, init_db, check_db_connection, db_manager
from .database.async_core import get_async_db, init_async_db, async_db_manager
from .database.models import User, Role, Item, AuditLog
from .database.crud import (
    crud_user, crud_item, crud_role, crud_audit_log,
    async_crud_user, async_crud_item
)

# Database dependencies for FastAPI
def get_database() -> Generator[Session, None, None]:
    """Get database session for dependency injection."""
    return get_db()

def get_async_database() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session for dependency injection."""
    return get_async_db()

# Database initialization
def setup_database():
    """Setup database on application startup."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Check database connection
        if check_db_connection():
            logger.info("Database connection established")
        else:
            logger.error("Failed to establish database connection")
            return False
        
        # Initialize database tables
        init_db()
        logger.info("Database tables initialized")
        
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {{e}}")
        return False

async def setup_async_database():
    """Setup async database on application startup."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize async database tables
        await init_async_db()
        logger.info("Async database tables initialized")
        
        return True
    except Exception as e:
        logger.error(f"Async database setup failed: {{e}}")
        return False

# Export commonly used components
__all__ = [
    # Database URLs
    "DATABASE_URL",
    "ASYNC_DATABASE_URL",
    "DB_CONFIG",
    
    # Database sessions
    "get_db",
    "get_async_db",
    "get_database",
    "get_async_database",
    
    # Database initialization
    "init_db",
    "init_async_db",
    "setup_database",
    "setup_async_database",
    "check_db_connection",
    
    # Database managers
    "db_manager",
    "async_db_manager",
    
    # Models
    "User",
    "Role", 
    "Item",
    "AuditLog",
    
    # CRUD operations
    "crud_user",
    "crud_item",
    "crud_role",
    "crud_audit_log",
    "async_crud_user",
    "async_crud_item"
]
'''
        return code

def main():
    """CLI interface for database setup."""
    parser = argparse.ArgumentParser(description="Setup FastAPI database configurations")
    parser.add_argument("--project-name", default="MyFastAPI", help="Project name")
    parser.add_argument("--database-url", default="sqlite:///./app.db", help="Database URL")
    parser.add_argument("--async-database-url", default="sqlite+aiosqlite:///./app.db", help="Async database URL")
    parser.add_argument("--output-dir", default="database", help="Output directory")
    parser.add_argument("--sqlalchemy", action="store_true", help="Generate SQLAlchemy configuration")
    parser.add_argument("--async-sqlalchemy", dest="async_sqlalchemy", action="store_true", help="Generate async SQLAlchemy configuration")
    parser.add_argument("--models", action="store_true", help="Generate database models")
    parser.add_argument("--crud", action="store_true", help="Generate CRUD operations")
    parser.add_argument("--migrations", action="store_true", help="Generate migrations template")
    parser.add_argument("--all", action="store_true", help="Generate all database configurations")
    
    args = parser.parse_args()
    
    setup = FastAPIDatabaseSetup()
    setup.database_url = args.database_url
    setup.async_database_url = args.async_database_url
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.all or args.sqlalchemy:
        core_file = output_dir / "core.py"
        core_file.write_text(setup.generate_sqlalchemy_config(args.project_name))
        print(f"SQLAlchemy configuration generated: {core_file}")
    
    if args.all or args.async_sqlalchemy:
        async_file = output_dir / "async_core.py"
        async_file.write_text(setup.generate_async_sqlalchemy_config())
        print(f"Async SQLAlchemy configuration generated: {async_file}")
    
    if args.all or args.models:
        models_file = output_dir / "models.py"
        models_file.write_text(setup.generate_database_models_template())
        print(f"Database models generated: {models_file}")
    
    if args.all or args.crud:
        crud_file = output_dir / "crud.py"
        crud_file.write_text(setup.generate_database_crud_operations())
        print(f"CRUD operations generated: {crud_file}")
    
    if args.all or args.migrations:
        migrations_file = output_dir / "migrations.py"
        migrations_file.write_text(setup.generate_migrations_template())
        print(f"Migrations template generated: {migrations_file}")
    
    if args.all:
        config_file = output_dir / "config.py"
        config_file.write_text(setup.generate_database_config_file())
        print(f"Database configuration generated: {config_file}")
    
    print("Database setup completed!")

if __name__ == "__main__":
    main()