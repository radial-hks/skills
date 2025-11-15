---
name: fastapi-unified
description: This skill should be used to build FastAPI applications with comprehensive knowledge of official documentation, async patterns, Pydantic models, dependencies, security, and deployment. This skill should be used when creating APIs, web services, or backend applications.
license: Complete terms in LICENSE.txt
---

# FastAPI Unified Skill

## Purpose

To build FastAPI applications with comprehensive knowledge of official documentation, async patterns, Pydantic models, path operations, dependencies, security implementations, database integrations, and deployment strategies. This skill provides the procedural knowledge needed to create production-ready FastAPI APIs and web services.

## When to Use This Skill

Use this skill when users need to:
- Create FastAPI applications from scratch
- Implement RESTful APIs with proper async patterns
- Design Pydantic models for request/response validation
- Configure dependencies and dependency injection
- Implement OAuth2 security and authentication
- Integrate with databases using SQLAlchemy
- Deploy FastAPI applications to production
- Debug FastAPI applications and resolve common issues

## How to Use This Skill

To build FastAPI applications, follow these steps:

1. **Set up project structure** using the provided templates
2. **Define Pydantic models** for request/response validation
3. **Create path operations** with proper HTTP methods and status codes
4. **Implement dependencies** for authentication, database connections, and shared logic
5. **Configure security** using OAuth2 and JWT tokens
6. **Add database integration** with SQLAlchemy and async support
7. **Test the application** using the provided testing utilities
8. **Deploy to production** using the deployment configurations

### Quick Start Implementation

To create a basic FastAPI application:

1. **Initialize project structure**:
```bash
mkdir my-fastapi-app
cd my-fastapi-app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn
```

2. **Create main application file**:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/items/")
def create_item(item: Item):
    return {"item": item}
```

3. **Run the application**:
```bash
uvicorn main:app --reload
```

### Advanced Implementation Patterns

To implement advanced FastAPI features:

1. **Use async path operations** for better performance:
   - Define async functions for I/O operations
   - Use async database connections
   - Implement proper error handling

2. **Implement dependency injection**:
   - Create reusable dependencies for authentication
   - Use dependencies for database sessions
   - Implement dependency caching

3. **Configure OAuth2 security**:
   - Set up OAuth2 password bearer
   - Implement JWT token generation
   - Create user authentication flows

### Using Bundled Resources

To leverage the bundled utilities:

1. **Use scripts/project_generator.py** for scaffolding:
   - Generate complete project structure
   - Create boilerplate code for common patterns
   - Set up configuration files

2. **Apply scripts/model_generator.py** for Pydantic models:
   - Generate models from database schemas
   - Create request/response models
   - Implement validation logic

3. **Implement security with scripts/security_setup.py**:
   - Configure OAuth2 authentication
   - Set up JWT token handling
   - Create user management systems

4. **Use scripts/database_setup.py** for database integration:
   - Configure SQLAlchemy connections
   - Set up async database operations
   - Create migration scripts

5. **Test applications with scripts/test_generator.py**:
   - Generate test cases for endpoints
   - Create integration tests
   - Set up test databases

6. **Deploy with scripts/deployment_config.py**:
   - Configure Docker containers
   - Set up production settings
   - Create deployment scripts

7. **Reference detailed documentation**:
   - references/async_patterns.md: Async programming patterns
   - references/pydantic_models.md: Model validation and serialization
   - references/dependencies_middleware.md: Dependency management and middleware patterns

### Best Practices Implementation

To ensure high-quality FastAPI applications:

1. **Follow RESTful design principles**:
   - Use appropriate HTTP methods
   - Return proper status codes
   - Implement consistent error handling

2. **Implement proper validation**:
   - Use Pydantic models for all inputs
   - Add custom validators when needed
   - Handle validation errors gracefully

3. **Optimize for performance**:
   - Use async operations for I/O
   - Implement proper database indexing
   - Use connection pooling

4. **Ensure security**:
   - Implement proper authentication
   - Use HTTPS in production
   - Validate all user inputs

### Common Use Case Implementations

To implement specific application types:

1. **CRUD API**:
   - Use SQLAlchemy models
   - Implement CRUD operations
   - Add pagination and filtering

2. **Authentication System**:
   - Set up OAuth2 password bearer
   - Implement user registration/login
   - Create protected endpoints

3. **File Upload API**:
   - Handle multipart uploads
   - Validate file types and sizes
   - Store files securely

4. **WebSocket Applications**:
   - Implement WebSocket endpoints
   - Handle real-time communication
   - Manage connection states

### Integration Patterns

To integrate FastAPI with external systems:

1. **Database Integration**:
   - Use SQLAlchemy for ORM
   - Implement async database operations
   - Handle connection management

2. **External API Integration**:
   - Use httpx for async HTTP requests
   - Implement proper error handling
   - Add circuit breakers for reliability

3. **Message Queue Integration**:
   - Use Celery or RQ for background tasks
   - Implement task queues
   - Handle task results

### Troubleshooting Implementation Issues

To resolve common problems:

1. **CORS issues**: Configure CORS middleware properly
2. **Async database errors**: Ensure proper async/await usage
3. **Pydantic validation errors**: Check model field types
4. **Dependency injection errors**: Verify dependency functions
5. **Performance issues**: Use profiling tools and optimize queries

For detailed implementation patterns and advanced configurations, see the reference materials in this skill.