#!/usr/bin/env python3
"""
FastAPI Test Generator Script

This script provides comprehensive testing utilities for FastAPI applications,
including unit tests, integration tests, API endpoint tests, and performance testing.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

class FastAPITestGenerator:
    """Generate test configurations and utilities for FastAPI applications."""
    
    def __init__(self):
        """Initialize test generator."""
        self.test_framework = "pytest"
        self.test_directory = "tests"
    
    def generate_pytest_config(self, project_name: str) -> str:
        """Generate pytest configuration."""
        
        code = f'''"""
Pytest Configuration for {project_name}

This module provides pytest configuration and test utilities for FastAPI applications.
"""

import pytest
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {{
    "base_url": "http://testserver",
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 1,
    "test_user_email": "test@example.com",
    "test_user_password": "testpassword123",
    "test_admin_email": "admin@example.com",
    "test_admin_password": "adminpassword123"
}}

# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API endpoint test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "database: mark test as database test"
    )
    config.addinivalue_line(
        "markers", "asyncio: mark test as async test"
    )

# Test fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_app():
    """Create test FastAPI application."""
    from main import app  # Import your FastAPI app
    return app

@pytest.fixture
def client(test_app) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(test_app) as client:
        yield client

@pytest.fixture
async def async_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=test_app, base_url=TEST_CONFIG["base_url"]) as ac:
        yield ac

@pytest.fixture(scope="session")
def test_user_token(client: TestClient) -> str:
    """Get test user token."""
    response = client.post(
        "/auth/login",
        data={{
            "username": TEST_CONFIG["test_user_email"],
            "password": TEST_CONFIG["test_user_password"]
        }}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        # Create test user if it doesn't exist
        client.post(
            "/auth/register",
            json={{
                "email": TEST_CONFIG["test_user_email"],
                "password": TEST_CONFIG["test_user_password"],
                "full_name": "Test User"
            }}
        )
        response = client.post(
            "/auth/login",
            data={{
                "username": TEST_CONFIG["test_user_email"],
                "password": TEST_CONFIG["test_user_password"]
            }}
        )
        return response.json()["access_token"]

@pytest.fixture(scope="session")
def test_admin_token(client: TestClient) -> str:
    """Get test admin token."""
    response = client.post(
        "/auth/login",
        data={{
            "username": TEST_CONFIG["test_admin_email"],
            "password": TEST_CONFIG["test_admin_password"]
        }}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        # Create test admin if it doesn't exist
        client.post(
            "/auth/register",
            json={{
                "email": TEST_CONFIG["test_admin_email"],
                "password": TEST_CONFIG["test_admin_password"],
                "full_name": "Test Admin",
                "is_superuser": True
            }}
        )
        response = client.post(
            "/auth/login",
            data={{
                "username": TEST_CONFIG["test_admin_email"],
                "password": TEST_CONFIG["test_admin_password"]
            }}
        )
        return response.json()["access_token"]

@pytest.fixture
def auth_headers(test_user_token: str) -> dict:
    """Get authentication headers for test user."""
    return {{"Authorization": f"Bearer {{test_user_token}}"}}

@pytest.fixture
def admin_headers(test_admin_token: str) -> dict:
    """Get authentication headers for test admin."""
    return {{"Authorization": f"Bearer {{test_admin_token}}"}}

@pytest.fixture
def api_key_headers() -> dict:
    """Get API key headers."""
    return {{"X-API-Key": "test-api-key-123"}}

# Database fixtures
@pytest.fixture
def test_db_session():
    """Create test database session."""
    from database import get_db
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def clean_database(test_db_session):
    """Clean database before each test."""
    # Add database cleanup logic here
    yield
    # Add database cleanup logic after test

# Test data fixtures
@pytest.fixture
def sample_user_data() -> dict:
    """Sample user data for testing."""
    return {{
        "username": "testuser",
        "email": "testuser@example.com",
        "full_name": "Test User",
        "password": "testpassword123"
    }}

@pytest.fixture
def sample_item_data() -> dict:
    """Sample item data for testing."""
    return {{
        "title": "Test Item",
        "description": "This is a test item",
        "price": 99.99,
        "is_available": True
    }}

@pytest.fixture
def sample_order_data() -> dict:
    """Sample order data for testing."""
    return {{
        "items": [1, 2, 3],
        "total_amount": 299.97,
        "shipping_address": {{
            "street": "123 Test St",
            "city": "Test City",
            "state": "TS",
            "zip": "12345"
        }}
    }}

# Utility functions
def assert_response_success(response, status_code: int = 200):
    """Assert that response was successful."""
    assert response.status_code == status_code, f"Expected {{status_code}}, got {{response.status_code}}: {{response.text}}"
    if response.status_code == 200:
        assert response.json().get("success", True) is True

def assert_response_error(response, status_code: int = 400):
    """Assert that response was an error."""
    assert response.status_code == status_code, f"Expected {{status_code}}, got {{response.status_code}}"
    if response.status_code >= 400:
        assert "error" in response.json() or "detail" in response.json()

def assert_valid_schema(response, expected_keys: list):
    """Assert response has valid schema."""
    data = response.json()
    if isinstance(data, dict):
        for key in expected_keys:
            assert key in data, f"Missing key '{{key}}' in response"
    elif isinstance(data, list):
        for item in data:
            for key in expected_keys:
                assert key in item, f"Missing key '{{key}}' in response item"

def retry_request(func, max_retries: int = None, delay: float = None):
    """Retry a request function."""
    max_retries = max_retries or TEST_CONFIG["max_retries"]
    delay = delay or TEST_CONFIG["retry_delay"]
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            import time
            time.sleep(delay)
    
    raise Exception("Max retries exceeded")

# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{{self.name}} took {{duration:.3f}} seconds")
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

def measure_performance(func, iterations: int = 10) -> dict:
    """Measure function performance."""
    import time
    
    durations = []
    for _ in range(iterations):
        start = time.time()
        func()
        end = time.time()
        durations.append(end - start)
    
    return {{
        "min": min(durations),
        "max": max(durations),
        "avg": sum(durations) / len(durations),
        "median": sorted(durations)[len(durations) // 2],
        "iterations": iterations
    }}

# Security testing utilities
def test_sql_injection(client: TestClient, endpoint: str, params: dict):
    """Test for SQL injection vulnerabilities."""
    malicious_params = {{}}
    for key, value in params.items():
        malicious_params[key] = f"{{value}}' OR '1'='1"
    
    response = client.get(endpoint, params=malicious_params)
    # Should not return all records or crash
    assert response.status_code != 500, "SQL injection may be possible"

def test_xss_injection(client: TestClient, endpoint: str, data: dict):
    """Test for XSS vulnerabilities."""
    xss_payload = "<script>alert('XSS')</script>"
    malicious_data = data.copy()
    
    for key in malicious_data:
        if isinstance(malicious_data[key], str):
            malicious_data[key] = f"{{malicious_data[key]}} {{xss_payload}}"
    
    response = client.post(endpoint, json=malicious_data)
    # Should sanitize input
    assert response.status_code != 500, "XSS may be possible"

# Test data generators
import random
import string

def generate_random_string(length: int = 10) -> str:
    """Generate random string."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_random_email() -> str:
    """Generate random email."""
    username = generate_random_string(8).lower()
    domain = random.choice(["example.com", "test.com", "demo.com"])
    return f"{{username}}@{{domain}}"

def generate_random_user() -> dict:
    """Generate random user data."""
    return {{
        "username": generate_random_string(8).lower(),
        "email": generate_random_email(),
        "full_name": f"Test {{generate_random_string(5)}} {{generate_random_string(7)}}",
        "password": generate_random_string(12)
    }}

def generate_random_item() -> dict:
    """Generate random item data."""
    return {{
        "title": f"Test {{generate_random_string(8)}}",
        "description": f"This is a test item: {{generate_random_string(20)}}",
        "price": round(random.uniform(10, 1000), 2),
        "is_available": random.choice([True, False])
    }}
'''
        return code
    
    def generate_test_config_file(self) -> str:
        """Generate main test configuration file."""
        
        code = '''"""
Test Configuration for FastAPI Application

This module provides comprehensive test setup for FastAPI applications.
"""

import pytest
from pathlib import Path
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_SETTINGS = {
    "test_database_url": "sqlite:///./test.db",
    "test_async_database_url": "sqlite+aiosqlite:///./test.db",
    "test_user_email": "test@example.com",
    "test_user_password": "testpassword123",
    "test_admin_email": "admin@example.com",
    "test_admin_password": "adminpassword123",
    "test_api_key": "test-api-key-123",
    "performance_test_iterations": 100,
    "load_test_duration_seconds": 30,
    "max_response_time_ms": 1000,
    "min_success_rate": 0.95
}

# Test discovery
pytest_plugins = [
    "tests.conftest",
    "tests.unit.test_services",
    "tests.unit.test_utils",
    "tests.integration.test_auth",
    "tests.integration.test_api",
    "tests.performance.test_benchmarks",
    "tests.security.test_vulnerabilities"
]

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add slow marker for performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker for integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add security marker for security tests
        if "security" in item.nodeid:
            item.add_marker(pytest.mark.security)

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "api: marks tests as API tests")
    config.addinivalue_line("markers", "database: marks tests as database tests")
    config.addinivalue_line("markers", "asyncio: marks tests as async tests")

# Test utilities
class TestHelpers:
    """Helper functions for tests."""
    
    @staticmethod
    def setup_test_database():
        """Setup test database."""
        from database import init_db, check_db_connection
        
        if check_db_connection():
            init_db()
            logger.info("Test database setup completed")
        else:
            logger.error("Failed to setup test database")
            raise Exception("Database connection failed")
    
    @staticmethod
    def cleanup_test_database():
        """Cleanup test database."""
        import os
        test_db_file = "test.db"
        if os.path.exists(test_db_file):
            os.remove(test_db_file)
            logger.info("Test database cleaned up")
    
    @staticmethod
    def create_test_user(client, user_data=None):
        """Create a test user."""
        if user_data is None:
            user_data = {
                "email": TEST_SETTINGS["test_user_email"],
                "username": "testuser",
                "password": TEST_SETTINGS["test_user_password"],
                "full_name": "Test User"
            }
        
        response = client.post("/auth/register", json=user_data)
        if response.status_code == 201:
            return response.json()
        elif response.status_code == 400:
            # User might already exist, try to login
            login_data = {
                "username": user_data["email"],
                "password": user_data["password"]
            }
            login_response = client.post("/auth/login", data=login_data)
            if login_response.status_code == 200:
                return login_response.json()
        
        return None
    
    @staticmethod
    def get_auth_headers(token):
        """Get authentication headers."""
        return {"Authorization": f"Bearer {token}"}
    
    @staticmethod
    def get_api_key_headers():
        """Get API key headers."""
        return {"X-API-Key": TEST_SETTINGS["test_api_key"]}

# Export commonly used functions
__all__ = [
    "TEST_SETTINGS",
    "TestHelpers",
    "pytest_configure",
    "pytest_collection_modifyitems"
]
'''
        return code

def main():
    """CLI interface for test setup."""
    parser = argparse.ArgumentParser(description="Setup FastAPI test configurations")
    parser.add_argument("--project-name", default="MyFastAPI", help="Project name")
    parser.add_argument("--output-dir", default="tests", help="Output directory")
    parser.add_argument("--pytest", action="store_true", help="Generate pytest configuration")
    parser.add_argument("--unit", action="store_true", help="Generate unit tests")
    parser.add_argument("--integration", action="store_true", help="Generate integration tests")
    parser.add_argument("--performance", action="store_true", help="Generate performance tests")
    parser.add_argument("--security", action="store_true", help="Generate security tests")
    parser.add_argument("--config", action="store_true", help="Generate test configuration")
    parser.add_argument("--all", action="store_true", help="Generate all test configurations")
    
    args = parser.parse_args()
    
    generator = FastAPITestGenerator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.all or args.pytest:
        conftest_file = output_dir / "conftest.py"
        conftest_file.write_text(generator.generate_pytest_config(args.project_name))
        print(f"Pytest configuration generated: {conftest_file}")
    
    if args.all or args.unit:
        unit_dir = output_dir / "unit"
        unit_dir.mkdir(exist_ok=True)
        
        unit_file = unit_dir / "test_services.py"
        unit_file.write_text(generator.generate_unit_tests_template())
        print(f"Unit tests generated: {unit_file}")
        
        # Create __init__.py
        (unit_dir / "__init__.py").write_text("")
    
    if args.all or args.integration:
        integration_dir = output_dir / "integration"
        integration_dir.mkdir(exist_ok=True)
        
        integration_file = integration_dir / "test_api.py"
        integration_file.write_text(generator.generate_integration_tests_template())
        print(f"Integration tests generated: {integration_file}")
        
        # Create __init__.py
        (integration_dir / "__init__.py").write_text("")
    
    if args.all or args.performance:
        performance_dir = output_dir / "performance"
        performance_dir.mkdir(exist_ok=True)
        
        performance_file = performance_dir / "test_benchmarks.py"
        performance_file.write_text(generator.generate_performance_tests_template())
        print(f"Performance tests generated: {performance_file}")
        
        # Create __init__.py
        (performance_dir / "__init__.py").write_text("")
    
    if args.all or args.security:
        security_dir = output_dir / "security"
        security_dir.mkdir(exist_ok=True)
        
        security_file = security_dir / "test_vulnerabilities.py"
        security_file.write_text(generator.generate_security_tests_template())
        print(f"Security tests generated: {security_file}")
        
        # Create __init__.py
        (security_dir / "__init__.py").write_text("")
    
    if args.all or args.config:
        config_file = output_dir / "config.py"
        config_file.write_text(generator.generate_test_config_file())
        print(f"Test configuration generated: {config_file}")
    
    # Create pytest.ini
    if args.all:
        pytest_ini_content = '''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    api: API endpoint tests
    performance: Performance tests
    security: Security tests
    database: Database tests
    asyncio: Async tests
    slow: Slow tests
'''
        pytest_ini_file = Path("pytest.ini")
        pytest_ini_file.write_text(pytest_ini_content)
        print(f"Pytest configuration file generated: {pytest_ini_file}")
    
    print("Test setup completed!")

if __name__ == "__main__":
    main()