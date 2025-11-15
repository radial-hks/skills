#!/usr/bin/env python3
"""
FastAPI Project Generator

This script generates complete FastAPI project structures with best practices,
including proper async patterns, Pydantic models, and project organization.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

class FastAPIProjectGenerator:
    """Generate FastAPI project structures with best practices."""
    
    def __init__(self):
        """Initialize the project generator."""
        self.project_templates = {
            "basic": self.generate_basic_project,
            "crud": self.generate_crud_project,
            "auth": self.generate_auth_project,
            "websocket": self.generate_websocket_project
        }
    
    def generate_basic_project(self, project_name: str, output_dir: str) -> Dict[str, str]:
        """Generate a basic FastAPI project structure."""
        
        files = {}
        
        # Main application file
        files[f"{project_name}/main.py"] = '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="{project_name}",
    description="A FastAPI application",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {{"message": "Hello World", "status": "running"}}

@app.get("/health")
def health_check():
    return {{"status": "healthy"}}
'''.format(project_name=project_name)
        
        # Requirements file
        files[f"{project_name}/requirements.txt"] = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
'''
        
        # Configuration file
        files[f"{project_name}/config.py"] = '''from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "{project_name}"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
'''.format(project_name=project_name)
        
        # Environment file
        files[f"{project_name}/.env.example"] = '''APP_NAME={project_name}
DEBUG=false
HOST=0.0.0.0
PORT=8000
RELOAD=false
'''.format(project_name=project_name)
        
        # Docker file
        files[f"{project_name}/Dockerfile"] = '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        # Docker Compose file
        files[f"{project_name}/docker-compose.yml"] = '''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
    volumes:
      - .:/app
'''
        
        return files
    
    def generate_crud_project(self, project_name: str, output_dir: str) -> Dict[str, str]:
        """Generate a CRUD FastAPI project with database integration."""
        
        files = self.generate_basic_project(project_name, output_dir)
        
        # Update requirements
        files[f"{project_name}/requirements.txt"] += '''sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9
asyncpg==0.29.0
'''
        
        # Database configuration
        files[f"{project_name}/database.py"] = '''from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
'''
        
        # Models
        files[f"{project_name}/models.py"] = '''from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from database import Base

class Item(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
'''
        
        # Pydantic schemas
        files[f"{project_name}/schemas.py"] = '''from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ItemBase(BaseModel):
    name: str
    description: Optional[str] = None

class ItemCreate(ItemBase):
    pass

class ItemUpdate(ItemBase):
    name: Optional[str] = None
    description: Optional[str] = None

class Item(ItemBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
'''
        
        # CRUD operations
        files[f"{project_name}/crud.py"] = '''from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import Item
from schemas import ItemCreate, ItemUpdate

async def create_item(db: AsyncSession, item: ItemCreate):
    db_item = Item(**item.dict())
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return db_item

async def get_items(db: AsyncSession, skip: int = 0, limit: int = 100):
    result = await db.execute(select(Item).offset(skip).limit(limit))
    return result.scalars().all()

async def get_item(db: AsyncSession, item_id: int):
    result = await db.execute(select(Item).where(Item.id == item_id))
    return result.scalar_one_or_none()

async def update_item(db: AsyncSession, item_id: int, item: ItemUpdate):
    db_item = await get_item(db, item_id)
    if db_item:
        update_data = item.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_item, field, value)
        await db.commit()
        await db.refresh(db_item)
    return db_item

async def delete_item(db: AsyncSession, item_id: int):
    db_item = await get_item(db, item_id)
    if db_item:
        await db.delete(db_item)
        await db.commit()
    return db_item
'''
        
        # Updated main.py with CRUD endpoints
        files[f"{project_name}/main.py"] = '''from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from database import get_db
from models import Item
from schemas import ItemCreate, ItemUpdate, Item
from crud import (
    create_item, get_items, get_item, 
    update_item, delete_item
)

app = FastAPI(
    title="{project_name}",
    description="A FastAPI CRUD application",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {{"message": "Hello World", "status": "running"}}

@app.get("/health")
def health_check():
    return {{"status": "healthy"}}

@app.post("/items/", response_model=Item)
async def create_new_item(item: ItemCreate, db: AsyncSession = Depends(get_db)):
    return await create_item(db, item)

@app.get("/items/", response_model=List[Item])
async def read_items(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    items = await get_items(db, skip=skip, limit=limit)
    return items

@app.get("/items/{{item_id}}", response_model=Item)
async def read_item(item_id: int, db: AsyncSession = Depends(get_db)):
    db_item = await get_item(db, item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@app.put("/items/{{item_id}}", response_model=Item)
async def update_existing_item(
    item_id: int, item: ItemUpdate, db: AsyncSession = Depends(get_db)
):
    db_item = await update_item(db, item_id=item_id, item=item)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@app.delete("/items/{{item_id}}", response_model=Item)
async def delete_existing_item(item_id: int, db: AsyncSession = Depends(get_db)):
    db_item = await delete_item(db, item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item
'''.format(project_name=project_name)
        
        return files
    
    def generate_auth_project(self, project_name: str, output_dir: str) -> Dict[str, str]:
        """Generate an authentication-enabled FastAPI project."""
        
        files = self.generate_crud_project(project_name, output_dir)
        
        # Update requirements
        files[f"{project_name}/requirements.txt"] += '''python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
'''
        
        # Authentication utilities
        files[f"{project_name}/auth.py"] = '''from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
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
        return username
    except JWTError:
        raise credentials_exception
'''
        
        return files
    
    def generate_websocket_project(self, project_name: str, output_dir: str) -> Dict[str, str]:
        """Generate a WebSocket-enabled FastAPI project."""
        
        files = self.generate_basic_project(project_name, output_dir)
        
        # WebSocket manager
        files[f"{project_name}/websocket_manager.py"] = '''from typing import List
from fastapi import WebSocket

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
'''
        
        # Updated main.py with WebSocket endpoints
        files[f"{project_name}/main.py"] = '''from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from websocket_manager import manager

app = FastAPI(
    title="{project_name}",
    description="A FastAPI WebSocket application",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {{"message": "Hello World", "status": "running"}}

@app.websocket("/ws/{{client_id}}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {{data}}", websocket)
            await manager.broadcast(f"Client {{client_id}} says: {{data}}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {{client_id}} left the chat")
'''.format(project_name=project_name)
        
        return files
    
    def create_project(self, project_type: str, project_name: str, output_dir: str = ".") -> bool:
        """Create a FastAPI project of the specified type."""
        
        if project_type not in self.project_templates:
            print(f"Available project types: {list(self.project_templates.keys())}")
            return False
        
        try:
            # Generate project files
            files = self.project_templates[project_type](project_name, output_dir)
            
            # Create directories and write files
            for file_path, content in files.items():
                full_path = Path(output_dir) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                print(f"Created: {file_path}")
            
            print(f"\nFastAPI {project_type} project '{project_name}' created successfully!")
            print(f"To run the application:")
            print(f"1. cd {project_name}")
            print(f"2. pip install -r requirements.txt")
            print(f"3. uvicorn main:app --reload")
            
            return True
            
        except Exception as e:
            print(f"Error creating project: {str(e)}")
            return False

def main():
    """CLI interface for the project generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate FastAPI project structures")
    parser.add_argument("type", choices=["basic", "crud", "auth", "websocket"], 
                       help="Type of FastAPI project to generate")
    parser.add_argument("name", help="Project name")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    generator = FastAPIProjectGenerator()
    generator.create_project(args.type, args.name, args.output)

if __name__ == "__main__":
    main()