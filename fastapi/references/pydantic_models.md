# FastAPI Pydantic Models Reference

This document provides comprehensive guidance on creating and using Pydantic models in FastAPI applications, covering validation, serialization, and advanced patterns.

## Table of Contents

1. [Basic Model Definition](#basic-model-definition)
2. [Field Validation](#field-validation)
3. [Custom Validators](#custom-validators)
4. [Model Inheritance](#model-inheritance)
5. [Nested Models](#nested-models)
6. [Database Integration](#database-integration)
7. [Serialization and Deserialization](#serialization-and-deserialization)
8. [Advanced Validation Patterns](#advanced-validation-patterns)
9. [Performance Optimization](#performance-optimization)
10. [Testing Models](#testing-models)

## Basic Model Definition

### Simple Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    """Base user model."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$', description="Valid email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    is_active: bool = Field(default=True, description="Account status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

class UserCreate(UserBase):
    """Model for creating users."""
    password: str = Field(..., min_length=8, max_length=100, description="User password")
    password_confirm: str = Field(..., description="Password confirmation")

class UserResponse(UserBase):
    """Model for user responses."""
    id: int = Field(..., description="User ID")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    class Config:
        orm_mode = True  # Enable ORM mode for SQLAlchemy compatibility
```

### Field Types and Constraints

```python
from pydantic import conint, confloat, constr
from typing import Union
from decimal import Decimal

class Product(BaseModel):
    """Product model with various field types."""
    # String constraints
    name: constr(min_length=1, max_length=100, strip_whitespace=True)
    description: Optional[constr(max_length=1000)] = None
    sku: constr(regex=r'^[A-Z]{3}-\d{4}$')  # e.g., ABC-1234
    
    # Numeric constraints
    price: confloat(gt=0, lt=10000)  # Greater than 0, less than 10000
    quantity: conint(ge=0, le=1000)   # Greater than or equal to 0, less than or equal to 1000
    rating: confloat(ge=0, le=5)      # Between 0 and 5
    
    # Union types
    identifier: Union[str, int]       # Can be either string or int
    
    # Decimal for financial calculations
    cost: Decimal = Field(..., decimal_places=2, ge=0)
    
    # Complex types
    tags: List[constr(min_length=1, max_length=20)] = []
    metadata: dict = Field(default_factory=dict)
```

## Field Validation

### Built-in Validators

```python
from pydantic import validator, root_validator

class UserRegistration(BaseModel):
    """User registration with validation."""
    email: str
    password: str
    password_confirm: str
    age: int
    phone: Optional[str] = None
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('age')
    def validate_age(cls, v):
        """Validate age."""
        if v < 18:
            raise ValueError('Must be at least 18 years old')
        if v > 120:
            raise ValueError('Invalid age')
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        """Validate phone number."""
        if v is None:
            return v
        # Remove non-digit characters
        phone_digits = ''.join(filter(str.isdigit, v))
        if len(phone_digits) != 10:
            raise ValueError('Phone number must be 10 digits')
        return phone_digits
    
    @root_validator
    def validate_passwords_match(cls, values):
        """Validate that passwords match."""
        password = values.get('password')
        password_confirm = values.get('password_confirm')
        if password != password_confirm:
            raise ValueError('Passwords do not match')
        return values
```

### Pre and Post Validators

```python
class Order(BaseModel):
    """Order model with pre and post validation."""
    items: List[str]
    quantities: List[int]
    total_amount: float
    
    @validator('quantities', pre=True)
    def parse_quantities(cls, v):
        """Pre-process quantities."""
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(',')]
        return v
    
    @validator('total_amount', pre=True)
    def parse_total_amount(cls, v):
        """Pre-process total amount."""
        if isinstance(v, str):
            return float(v.replace('$', '').replace(',', ''))
        return v
    
    @validator('total_amount')
    def validate_total_amount(cls, v, values):
        """Post-validate total amount."""
        items = values.get('items', [])
        quantities = values.get('quantities', [])
        
        if len(items) != len(quantities):
            raise ValueError('Items and quantities must have the same length')
        
        if v <= 0:
            raise ValueError('Total amount must be positive')
        
        return round(v, 2)
```

## Custom Validators

### Advanced Validation Logic

```python
from typing import Any, Dict
import re

class BusinessRegistration(BaseModel):
    """Business registration with complex validation."""
    business_name: str
    tax_id: str
    website: Optional[str] = None
    business_type: str
    founding_date: str
    employees: int
    annual_revenue: float
    
    @validator('business_name')
    def validate_business_name(cls, v):
        """Validate business name."""
        if len(v) < 2:
            raise ValueError('Business name must be at least 2 characters')
        if len(v) > 100:
            raise ValueError('Business name must be at most 100 characters')
        if not re.match(r'^[\w\s\-&.,]+$', v):
            raise ValueError('Business name contains invalid characters')
        return v.strip()
    
    @validator('tax_id')
    def validate_tax_id(cls, v):
        """Validate tax ID (EIN format: XX-XXXXXXX)."""
        if not re.match(r'^\d{2}-\d{7}$', v):
            raise ValueError('Tax ID must be in format XX-XXXXXXX')
        return v
    
    @validator('website')
    def validate_website(cls, v):
        """Validate website URL."""
        if v is None:
            return v
        
        if not re.match(r'^https?://[\w\-]+(\.[\w\-]+)+[/#?]?.*$', v):
            raise ValueError('Invalid website URL')
        return v
    
    @validator('business_type')
    def validate_business_type(cls, v):
        """Validate business type."""
        valid_types = ['LLC', 'Corporation', 'Partnership', 'Sole Proprietorship']
        if v not in valid_types:
            raise ValueError(f'Business type must be one of: {valid_types}')
        return v
    
    @validator('founding_date')
    def validate_founding_date(cls, v):
        """Validate founding date."""
        try:
            date = datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Founding date must be in YYYY-MM-DD format')
        
        if date > datetime.now():
            raise ValueError('Founding date cannot be in the future')
        
        if date < datetime(1900, 1, 1):
            raise ValueError('Founding date cannot be before 1900')
        
        return v
    
    @validator('employees')
    def validate_employees(cls, v):
        """Validate employee count."""
        if v < 0:
            raise ValueError('Employee count cannot be negative')
        if v > 1000000:
            raise ValueError('Employee count seems unrealistic')
        return v
    
    @validator('annual_revenue')
    def validate_annual_revenue(cls, v):
        """Validate annual revenue."""
        if v < 0:
            raise ValueError('Annual revenue cannot be negative')
        if v > 1000000000000:  # 1 trillion
            raise ValueError('Annual revenue seems unrealistic')
        return round(v, 2)
```

### Cross-Field Validation

```python
class Address(BaseModel):
    """Address model with cross-field validation."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"
    
    @validator('state')
    def validate_state(cls, v, values):
        """Validate state based on country."""
        country = values.get('country', 'USA')
        
        if country == 'USA':
            valid_states = [
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
            ]
            if v.upper() not in valid_states:
                raise ValueError(f'Invalid US state: {v}')
        
        return v.upper()
    
    @validator('zip_code')
    def validate_zip_code(cls, v, values):
        """Validate ZIP code based on country."""
        country = values.get('country', 'USA')
        
        if country == 'USA':
            if not re.match(r'^\d{5}(-\d{4})?$', v):
                raise ValueError('US ZIP code must be in format XXXXX or XXXXX-XXXX')
        
        return v
    
    @root_validator
    def validate_address_completeness(cls, values):
        """Validate address completeness."""
        street = values.get('street')
        city = values.get('city')
        state = values.get('state')
        zip_code = values.get('zip_code')
        
        if not all([street, city, state, zip_code]):
            raise ValueError('All address fields are required')
        
        return values
```

## Model Inheritance

### Base Model Patterns

```python
class BaseModelConfig:
    """Base configuration for all models."""
    
    class Config:
        orm_mode = True
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v),
        }

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return v or datetime.utcnow()

class SoftDeleteMixin(BaseModel):
    """Mixin for soft delete functionality."""
    is_deleted: bool = Field(default=False)
    deleted_at: Optional[datetime] = None
    
    def soft_delete(self):
        """Mark as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self):
        """Restore from soft delete."""
        self.is_deleted = False
        self.deleted_at = None
```

### Inheritance Examples

```python
class UserBase(TimestampMixin, SoftDeleteMixin, BaseModelConfig):
    """Base user model with common fields."""
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True

class UserCreate(UserBase):
    """Model for creating users."""
    password: str
    password_confirm: str
    
    @validator('password')
    def validate_password(cls, v):
        # Password validation logic
        return v

class UserUpdate(BaseModelConfig):
    """Model for updating users."""
    username: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    
    @root_validator
    def validate_at_least_one_field(cls, values):
        if not any(values.values()):
            raise ValueError('At least one field must be provided for update')
        return values

class UserResponse(UserBase):
    """Model for user responses."""
    id: int
    last_login: Optional[datetime] = None
    
    class Config(UserBase.Config):
        fields = {'password': {'exclude': True}}  # Exclude password from responses
```

## Nested Models

### Complex Nested Structures

```python
class Address(BaseModel):
    """Address model."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"

class Company(BaseModel):
    """Company model."""
    name: str
    tax_id: str
    address: Address
    website: Optional[str] = None
    
    @validator('tax_id')
    def validate_tax_id(cls, v):
        # Tax ID validation
        return v

class UserProfile(BaseModel):
    """User profile with nested models."""
    user: UserResponse
    company: Optional[Company] = None
    addresses: List[Address] = []
    preferences: dict = Field(default_factory=dict)
    
    @validator('addresses')
    def validate_addresses(cls, v):
        """Validate address list."""
        if len(v) > 5:
            raise ValueError('Maximum 5 addresses allowed')
        return v
    
    @validator('preferences')
    def validate_preferences(cls, v):
        """Validate preferences dictionary."""
        allowed_keys = ['theme', 'language', 'notifications', 'privacy']
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f'Invalid preference key: {key}')
        return v
```

### Self-Referential Models

```python
class Category(BaseModel):
    """Category model with self-reference."""
    id: int
    name: str
    parent_id: Optional[int] = None
    children: List['Category'] = []
    
    class Config:
        orm_mode = True

# Forward reference for self-referential model
Category.update_forward_refs()

class Comment(BaseModel):
    """Comment model with self-reference."""
    id: int
    content: str
    author: UserResponse
    parent_id: Optional[int] = None
    replies: List['Comment'] = []
    created_at: datetime
    
    class Config:
        orm_mode = True

Comment.update_forward_refs()
```

## Database Integration

### SQLAlchemy Integration

```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

class UserDB(Base):
    """SQLAlchemy User model."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    posts = relationship("PostDB", back_populates="author")
    profile = relationship("UserProfileDB", back_populates="user", uselist=False)

class UserProfileDB(Base):
    """SQLAlchemy UserProfile model."""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    bio = Column(String(500), nullable=True)
    website = Column(String(200), nullable=True)
    location = Column(String(100), nullable=True)
    birth_date = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("UserDB", back_populates="profile")
```

### Model Conversion Utilities

```python
def sqlalchemy_to_pydantic(db_model: Base, pydantic_model: Type[BaseModel]) -> BaseModel:
    """Convert SQLAlchemy model to Pydantic model."""
    return pydantic_model.from_orm(db_model)

def pydantic_to_sqlalchemy(pydantic_model: BaseModel, db_model_class: Type[Base]) -> Base:
    """Convert Pydantic model to SQLAlchemy model."""
    return db_model_class(**pydantic_model.dict())

class ModelConverter:
    """Utility class for model conversions."""
    
    @staticmethod
    def user_db_to_response(user_db: UserDB) -> UserResponse:
        """Convert UserDB to UserResponse."""
        return UserResponse.from_orm(user_db)
    
    @staticmethod
    def user_create_to_db(user_create: UserCreate, hashed_password: str) -> UserDB:
        """Convert UserCreate to UserDB."""
        return UserDB(
            username=user_create.username,
            email=user_create.email,
            full_name=user_create.full_name,
            hashed_password=hashed_password,
            is_active=user_create.is_active
        )
```

## Serialization and Deserialization

### Custom Serialization

```python
from pydantic import BaseConfig
import json

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for special types."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

class CustomModel(BaseModel):
    """Model with custom serialization."""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v),
            UUID: lambda v: str(v),
        }
    
    def json_custom(self, **kwargs):
        """Custom JSON serialization."""
        return json.dumps(self.dict(), cls=CustomJSONEncoder, **kwargs)
    
    def dict_custom(self, **kwargs):
        """Custom dictionary serialization."""
        data = self.dict(**kwargs)
        # Custom processing
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Decimal):
                data[key] = str(value)
        return data
```

### Dynamic Model Creation

```python
def create_dynamic_model(fields: Dict[str, Any], model_name: str = "DynamicModel") -> Type[BaseModel]:
    """Create dynamic Pydantic model."""
    return type(model_name, (BaseModel,), fields)

# Usage
DynamicUserModel = create_dynamic_model({
    'name': (str, ...),
    'age': (int, Field(default=18, ge=0, le=120)),
    'email': (str, Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$'))
})

# Create instance
dynamic_user = DynamicUserModel(name="John", email="john@example.com")
```

## Advanced Validation Patterns

### Conditional Validation

```python
from typing import Literal

class Payment(BaseModel):
    """Payment model with conditional validation."""
    amount: float
    currency: Literal["USD", "EUR", "GBP"]
    method: Literal["credit_card", "bank_transfer", "paypal"]
    
    # Conditional fields
    card_number: Optional[str] = None
    expiry_date: Optional[str] = None
    cvv: Optional[str] = None
    
    bank_account: Optional[str] = None
    routing_number: Optional[str] = None
    
    paypal_email: Optional[str] = None
    
    @validator('card_number', 'expiry_date', 'cvv')
    def validate_credit_card_fields(cls, v, values, field):
        """Validate credit card fields when method is credit_card."""
        method = values.get('method')
        if method == 'credit_card' and not v:
            raise ValueError(f'{field.name} is required for credit card payments')
        return v
    
    @validator('bank_account', 'routing_number')
    def validate_bank_fields(cls, v, values, field):
        """Validate bank fields when method is bank_transfer."""
        method = values.get('method')
        if method == 'bank_transfer' and not v:
            raise ValueError(f'{field.name} is required for bank transfer payments')
        return v
    
    @validator('paypal_email')
    def validate_paypal_fields(cls, v, values):
        """Validate PayPal fields when method is paypal."""
        method = values.get('method')
        if method == 'paypal' and not v:
            raise ValueError('PayPal email is required for PayPal payments')
        return v
    
    @validator('card_number')
    def validate_card_number(cls, v):
        """Validate credit card number using Luhn algorithm."""
        if v is None:
            return v
        
        # Remove spaces and dashes
        card_number = v.replace(' ', '').replace('-', '')
        
        # Check if all digits
        if not card_number.isdigit():
            raise ValueError('Card number must contain only digits')
        
        # Luhn algorithm
        def luhn_checksum(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10
        
        if luhn_checksum(card_number) != 0:
            raise ValueError('Invalid credit card number')
        
        return card_number
```

### Enum Validation

```python
from enum import Enum
from pydantic import BaseModel

class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"
    GUEST = "guest"

class UserStatus(str, Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class UserWithEnum(BaseModel):
    """User model with enum validation."""
    username: str
    email: str
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.PENDING
    permissions: List[str] = []
    
    @validator('permissions', pre=True)
    def validate_permissions(cls, v, values):
        """Validate permissions based on role."""
        role = values.get('role')
        if isinstance(role, str):
            role = UserRole(role)
        
        # Default permissions based on role
        role_permissions = {
            UserRole.ADMIN: ['read', 'write', 'delete', 'admin'],
            UserRole.MODERATOR: ['read', 'write', 'moderate'],
            UserRole.USER: ['read', 'write'],
            UserRole.GUEST: ['read']
        }
        
        if not v:  # If no permissions provided, use defaults
            return role_permissions.get(role, [])
        
        # Validate provided permissions
        allowed_permissions = role_permissions.get(role, [])
        for permission in v:
            if permission not in allowed_permissions:
                raise ValueError(f'Permission {permission} not allowed for role {role}')
        
        return v
```

## Performance Optimization

### Model Configuration

```python
class OptimizedModel(BaseModel):
    """Optimized model with performance settings."""
    
    class Config:
        # Performance optimizations
        validate_assignment = False  # Skip validation on assignment
        use_enum_values = True       # Use enum values instead of instances
        allow_population_by_field_name = True  # Allow population by field name
        
        # Memory optimizations
        keep_untouched = (cached_property,)  # Keep specific types untouched
        
        # Serialization optimizations
        json_encoders = {
            datetime: lambda v: v.timestamp(),  # Use timestamp for datetime
        }
    
    # Use slots for memory efficiency
    __slots__ = ('_cache',)
    
    def __init__(__pydantic_self__, **data):
        super().__init__(**data)
        __pydantic_self__._cache = {}
```

### Lazy Validation

```python
from pydantic import BaseModel, Field
from typing import Any, Dict

class LazyValidationModel(BaseModel):
    """Model with lazy validation."""
    data: Dict[str, Any] = Field(default_factory=dict)
    
    def validate_field(self, field_name: str, value: Any):
        """Validate individual field lazily."""
        # Custom validation logic
        if field_name == 'email':
            if '@' not in value:
                raise ValueError('Invalid email format')
        return value
    
    def validate_all(self):
        """Validate all fields."""
        for field_name, value in self.data.items():
            self.validate_field(field_name, value)

# Usage
lazy_model = LazyValidationModel()
lazy_model.data['email'] = 'test@example.com'
lazy_model.validate_field('email', lazy_model.data['email'])
```

## Testing Models

### Model Validation Tests

```python
import pytest
from pydantic import ValidationError

class TestUserModels:
    """Test cases for user models."""
    
    def test_user_create_valid(self):
        """Test valid user creation."""
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123',
            'password_confirm': 'TestPassword123'
        }
        user = UserCreate(**user_data)
        assert user.username == 'testuser'
        assert user.email == 'test@example.com'
    
    def test_user_create_invalid_email(self):
        """Test invalid email validation."""
        user_data = {
            'username': 'testuser',
            'email': 'invalid-email',
            'password': 'TestPassword123',
            'password_confirm': 'TestPassword123'
        }
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(**user_data)
        assert 'Invalid email format' in str(exc_info.value)
    
    def test_user_create_password_mismatch(self):
        """Test password mismatch validation."""
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPassword123',
            'password_confirm': 'DifferentPassword'
        }
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(**user_data)
        assert 'Passwords do not match' in str(exc_info.value)
    
    def test_user_create_weak_password(self):
        """Test weak password validation."""
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'weak',
            'password_confirm': 'weak'
        }
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(**user_data)
        assert 'at least 8 characters' in str(exc_info.value)
```

### Model Serialization Tests

```python
def test_user_response_serialization(self):
    """Test user response serialization."""
    user_response = UserResponse(
        id=1,
        username='testuser',
        email='test@example.com',
        created_at=datetime(2023, 1, 1, 12, 0, 0)
    )
    
    # Test dictionary serialization
    user_dict = user_response.dict()
    assert user_dict['id'] == 1
    assert user_dict['username'] == 'testuser'
    
    # Test JSON serialization
    user_json = user_response.json()
    assert '"id": 1' in user_json
    assert '"username": "testuser"' in user_json

def test_nested_model_serialization(self):
    """Test nested model serialization."""
    address = Address(
        street='123 Main St',
        city='Anytown',
        state='CA',
        zip_code='12345'
    )
    
    company = Company(
        name='Test Company',
        tax_id='12-3456789',
        address=address
    )
    
    company_dict = company.dict()
    assert company_dict['name'] == 'Test Company'
    assert company_dict['address']['city'] == 'Anytown'
```

## Best Practices

1. **Use type hints**: Always use proper type hints for better IDE support and validation
2. **Field descriptions**: Add descriptions to fields for better documentation
3. **Validation logic**: Keep validation logic simple and focused
4. **Error messages**: Provide clear, user-friendly error messages
5. **Model inheritance**: Use inheritance to avoid code duplication
6. **ORM mode**: Enable `orm_mode` for database integration
7. **Performance**: Use model configuration for performance optimization
8. **Testing**: Write comprehensive tests for model validation
9. **Documentation**: Document complex validation logic
10. **Versioning**: Version your models when making breaking changes