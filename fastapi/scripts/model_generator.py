#!/usr/bin/env python3
"""
FastAPI Model Generator

This script generates Pydantic models for FastAPI applications from various sources,
including database schemas, JSON examples, and custom specifications.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from pathlib import Path

class FastAPIModelGenerator:
    """Generate Pydantic models for FastAPI applications."""
    
    def __init__(self):
        """Initialize the model generator."""
        self.type_mapping = {
            'str': 'str',
            'string': 'str',
            'int': 'int',
            'integer': 'int',
            'float': 'float',
            'number': 'float',
            'bool': 'bool',
            'boolean': 'bool',
            'datetime': 'datetime',
            'date': 'date',
            'list': 'List',
            'dict': 'Dict',
            'object': 'Dict[str, Any]'
        }
    
    def generate_from_json(self, json_data: Union[str, Dict], class_name: str = "GeneratedModel") -> str:
        """Generate Pydantic model from JSON data."""
        
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        else:
            data = json_data
        
        if not isinstance(data, dict):
            raise ValueError("JSON data must be a dictionary/object")
        
        fields = self._analyze_dict(data)
        return self._generate_model_code(fields, class_name)
    
    def generate_from_schema(self, schema: Dict, class_name: str = "GeneratedModel") -> str:
        """Generate Pydantic model from JSON Schema."""
        
        if 'properties' not in schema:
            raise ValueError("Schema must contain 'properties' field")
        
        fields = []
        required_fields = schema.get('required', [])
        
        for field_name, field_spec in schema['properties'].items():
            field_info = self._parse_json_schema_field(field_name, field_spec, field_name in required_fields)
            fields.append(field_info)
        
        return self._generate_model_code(fields, class_name)
    
    def generate_from_sqlalchemy(self, table_name: str, columns: List[Dict]) -> str:
        """Generate Pydantic models from SQLAlchemy table definition."""
        
        # Create base model
        base_fields = []
        create_fields = []
        update_fields = []
        
        for column in columns:
            field_name = column['name']
            field_type = self._map_sqlalchemy_type(column['type'])
            nullable = column.get('nullable', True)
            default = column.get('default', None)
            primary_key = column.get('primary_key', False)
            autoincrement = column.get('autoincrement', False)
            server_default = column.get('server_default', False)
            
            # Base model field (always included)
            base_field = {
                'name': field_name,
                'type': field_type,
                'optional': nullable and not primary_key,
                'default': default,
                'description': column.get('description', '')
            }
            base_fields.append(base_field)
            
            # Create model field (excludes auto-generated fields)
            if not autoincrement and not server_default and not primary_key:
                create_field = base_field.copy()
                create_fields.append(create_field)
            elif primary_key and not autoincrement and not server_default:
                create_field = base_field.copy()
                create_fields.append(create_field)
            
            # Update model field (all fields optional)
            update_field = base_field.copy()
            update_field['optional'] = True
            update_fields.append(update_field)
        
        # Generate model code
        code = "# Pydantic models generated from SQLAlchemy table\n\n"
        code += "from pydantic import BaseModel, Field\n"
        code += "from typing import Optional\n"
        code += "from datetime import datetime\n"
        code += "from decimal import Decimal\n\n"
        
        # Base model
        code += self._generate_model_code(base_fields, f"{table_name.title()}")
        code += "\n\n"
        
        # Create model
        if create_fields:
            code += self._generate_model_code(create_fields, f"{table_name.title()}Create")
            code += "\n\n"
        
        # Update model
        if update_fields:
            code += self._generate_model_code(update_fields, f"{table_name.title()}Update")
            code += "\n\n"
        
        return code
    
    def generate_crud_models(self, model_name: str, fields: List[Dict]) -> str:
        """Generate complete CRUD model set for FastAPI."""
        
        code = f"""# CRUD Pydantic models for {model_name}

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class {model_name}Base(BaseModel):
"""
        
        # Base model fields
        for field in fields:
            field_line = f"    {field['name']}: "
            
            if field.get('optional'):
                field_line += f"Optional[{field['type']}]"
                if field.get('default') is not None:
                    field_line += f" = {field['default']}"
                else:
                    field_line += " = None"
            else:
                field_line += field['type']
                if field.get('default') is not None:
                    field_line += f" = {field['default']}"
            
            if field.get('description'):
                field_line += f"  # {field['description']}"
            
            code += field_line + "\n"
        
        # Create model
        code += f"""
class {model_name}Create({model_name}Base):
    pass

class {model_name}Update(BaseModel):
"""
        
        # Update model fields (all optional)
        for field in fields:
            code += f"    {field['name']}: Optional[{field['type']}] = None"
            if field.get('description'):
                code += f"  # {field['description']}"
            code += "\n"
        
        # Response model
        code += f"""
class {model_name}({model_name}Base):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class {model_name}List(BaseModel):
    items: List[{model_name}]
    total: int
    skip: int
    limit: int
"""
        
        return code
    
    def generate_api_response_models(self, resource_name: str) -> str:
        """Generate standard API response models."""
        
        code = f"""# API Response models for {resource_name}

from pydantic import BaseModel
from typing import Generic, TypeVar, Optional

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    message: Optional[str] = None
    errors: Optional[list] = None

class {resource_name}Response(BaseModel):
    message: str
    status: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[dict] = None
    timestamp: str

class ValidationErrorResponse(BaseModel):
    error: str
    validation_errors: dict
    timestamp: str
"""
        
        return code
    
    def _analyze_dict(self, data: Dict, prefix: str = "") -> List[Dict]:
        """Analyze dictionary structure to determine field types."""
        
        fields = []
        
        for key, value in data.items():
            field_info = {
                'name': key,
                'type': self._infer_type(value),
                'optional': False,
                'default': None,
                'description': ''
            }
            
            # Handle None values
            if value is None:
                field_info['optional'] = True
                field_info['default'] = None
            
            fields.append(field_info)
        
        return fields
    
    def _infer_type(self, value: Any) -> str:
        """Infer Python type from JSON value."""
        
        if value is None:
            return 'None'
        elif isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            # Try to infer more specific types
            if self._is_datetime(value):
                return 'datetime'
            elif self._is_date(value):
                return 'date'
            elif self._is_email(value):
                return 'EmailStr'
            else:
                return 'str'
        elif isinstance(value, list):
            if len(value) > 0:
                item_type = self._infer_type(value[0])
                return f'List[{item_type}]'
            else:
                return 'List'
        elif isinstance(value, dict):
            return 'Dict[str, Any]'
        else:
            return 'Any'
    
    def _is_datetime(self, value: str) -> bool:
        """Check if string is a datetime."""
        datetime_patterns = [
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
        ]
        
        for pattern in datetime_patterns:
            if re.match(pattern, value):
                return True
        return False
    
    def _is_date(self, value: str) -> bool:
        """Check if string is a date."""
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',
            r'^\d{2}/\d{2}/\d{4}$',
            r'^\d{4}/\d{2}/\d{2}$',
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        return False
    
    def _is_email(self, value: str) -> bool:
        """Check if string is an email address."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, value) is not None
    
    def _parse_json_schema_field(self, field_name: str, field_spec: Dict, required: bool) -> Dict:
        """Parse JSON Schema field specification."""
        
        field_type = field_spec.get('type', 'string')
        field_format = field_spec.get('format', '')
        
        # Map JSON Schema types to Python types
        if field_format == 'date-time':
            py_type = 'datetime'
        elif field_format == 'date':
            py_type = 'date'
        elif field_format == 'email':
            py_type = 'EmailStr'
        elif field_format == 'uri':
            py_type = 'HttpUrl'
        elif field_type == 'array':
            items = field_spec.get('items', {{}})
            item_type = self._map_json_schema_type(items.get('type', 'string'))
            py_type = f'List[{item_type}]'
        elif field_type == 'object':
            py_type = 'Dict[str, Any]'
        else:
            py_type = self._map_json_schema_type(field_type)
        
        return {
            'name': field_name,
            'type': py_type,
            'optional': not required,
            'default': field_spec.get('default'),
            'description': field_spec.get('description', '')
        }
    
    def _map_json_schema_type(self, json_type: str) -> str:
        """Map JSON Schema type to Python type."""
        
        type_map = {
            'string': 'str',
            'number': 'float',
            'integer': 'int',
            'boolean': 'bool',
            'array': 'List',
            'object': 'Dict[str, Any]',
            'null': 'None'
        }
        
        return type_map.get(json_type, 'Any')
    
    def _map_sqlalchemy_type(self, sql_type: str) -> str:
        """Map SQLAlchemy type to Python type."""
        
        sql_map = {
            'Integer': 'int',
            'String': 'str',
            'Text': 'str',
            'Boolean': 'bool',
            'DateTime': 'datetime',
            'Date': 'date',
            'Float': 'float',
            'Numeric': 'Decimal',
            'JSON': 'Dict[str, Any]'
        }
        
        return sql_map.get(sql_type, 'Any')
    
    def generate_from_openapi(self, openapi_spec: Dict, component_name: str = None) -> str:
        """Generate Pydantic models from OpenAPI specification."""
        
        if 'components' not in openapi_spec or 'schemas' not in openapi_spec['components']:
            raise ValueError("OpenAPI spec must contain components.schemas")
        
        schemas = openapi_spec['components']['schemas']
        
        if component_name:
            if component_name not in schemas:
                raise ValueError(f"Component '{component_name}' not found in OpenAPI spec")
            return self.generate_from_schema(schemas[component_name], component_name)
        
        # Generate all models
        all_models = []
        for schema_name, schema_def in schemas.items():
            try:
                model_code = self.generate_from_schema(schema_def, schema_name)
                all_models.append(model_code)
            except Exception as e:
                print(f"Warning: Skipping schema '{schema_name}': {str(e)}")
        
        return "\n\n".join(all_models)
    
    def generate_validation_models(self, base_model: str, validation_rules: Dict) -> str:
        """Generate models with custom validation rules."""
        
        code = f"""# Pydantic models with custom validation

from pydantic import BaseModel, Field, validator, EmailStr, HttpUrl
from typing import Optional, List
from datetime import datetime

class {base_model}(BaseModel):
"""
        
        # Add fields with validation
        for field_name, rules in validation_rules.items():
            field_type = rules.get('type', 'str')
            optional = rules.get('optional', False)
            description = rules.get('description', '')
            
            if optional:
                code += f"    {field_name}: Optional[{field_type}] = None"
            else:
                code += f"    {field_name}: {field_type}"
            
            if description:
                code += f"  # {description}"
            code += "\n"
        
        # Add validators
        for field_name, rules in validation_rules.items():
            if 'validate' in rules:
                validator_name = f"validate_{field_name}"
                code += f"\n    @validator('{field_name}')\n"
                code += f"    def {validator_name}(cls, v):\n"
                code += f"        # Custom validation for {field_name}\n"
                code += f"        if v is not None:\n"
                code += f"            # Add validation logic here\n"
                code += f"            pass\n"
                code += f"        return v\n"
        
        return code
    
    def generate_enum_models(self, enums: Dict[str, List[str]]) -> str:
        """Generate enum models for FastAPI."""
        
        code = "# Enum models for FastAPI\n\n"
        code += "from enum import Enum\n"
        code += "from pydantic import BaseModel\n"
        code += "from typing import Optional\n\n"
        
        for enum_name, values in enums.items():
            code += f"class {enum_name}(str, Enum):\n"
            for value in values:
                code += f"    {value.upper()} = '{value}'\n"
            code += "\n"
        
        return code
    
    def _generate_model_code(self, fields: List[Dict], class_name: str) -> str:
        """Generate Pydantic model code from field definitions."""
        
        code = f"class {class_name}(BaseModel):\n"
        
        for field in fields:
            field_line = f"    {field['name']}: "
            
            if field.get('optional'):
                field_line += f"Optional[{field['type']}]"
                if field.get('default') is not None:
                    field_line += f" = {field['default']}"
                else:
                    field_line += " = None"
            else:
                field_line += field['type']
                if field.get('default') is not None:
                    field_line += f" = {field['default']}"
            
            if field.get('description'):
                field_line += f"  # {field['description']}"
            
            code += field_line + "\n"
        
        return code

def main():
    """CLI interface for the model generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Pydantic models for FastAPI")
    parser.add_argument("input", help="Input JSON file or string")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--class-name", "-c", default="GeneratedModel", help="Class name for the model")
    parser.add_argument("--type", "-t", choices=["json", "schema", "crud", "api", "openapi", "validation", "enum", "sqlalchemy"], 
                       default="json", help="Input type")
    parser.add_argument("--component", help="Specific OpenAPI component to generate")
    
    args = parser.parse_args()
    
    generator = FastAPIModelGenerator()
    
    try:
        if args.type == "json":
            # Try to load as file first, then as JSON string
            try:
                with open(args.input, 'r') as f:
                    json_data = f.read()
            except FileNotFoundError:
                json_data = args.input
            
            result = generator.generate_from_json(json_data, args.class_name)
        
        elif args.type == "schema":
            with open(args.input, 'r') as f:
                schema = json.load(f)
            result = generator.generate_from_schema(schema, args.class_name)
        
        elif args.type == "crud":
            # Example CRUD model generation
            fields = [
                {"name": "name", "type": "str", "optional": False},
                {"name": "description", "type": "str", "optional": True},
                {"name": "price", "type": "float", "optional": False},
                {"name": "is_active", "type": "bool", "optional": True, "default": True}
            ]
            result = generator.generate_crud_models(args.class_name, fields)
        
        elif args.type == "api":
            result = generator.generate_api_response_models(args.class_name)
        
        elif args.type == "openapi":
            with open(args.input, 'r') as f:
                openapi_spec = json.load(f)
            result = generator.generate_from_openapi(openapi_spec, args.component)
        
        elif args.type == "validation":
            # Example validation rules
            validation_rules = {
                "email": {"type": "EmailStr", "optional": False, "validate": True},
                "age": {"type": "int", "optional": True, "validate": True},
                "website": {"type": "HttpUrl", "optional": True}
            }
            result = generator.generate_validation_models(args.class_name, validation_rules)
        
        elif args.type == "enum":
            # Example enum generation
            enums = {
                "UserRole": ["admin", "user", "guest"],
                "Status": ["active", "inactive", "pending"]
            }
            result = generator.generate_enum_models(enums)
        
        elif args.type == "sqlalchemy":
            # Example SQLAlchemy table definition
            columns = [
                {"name": "id", "type": "Integer", "primary_key": True, "autoincrement": True},
                {"name": "name", "type": "String", "nullable": False},
                {"name": "email", "type": "String", "nullable": False, "default": None},
                {"name": "created_at", "type": "DateTime", "server_default": True}
            ]
            result = generator.generate_from_sqlalchemy(args.class_name.lower(), columns)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result)
            print(f"Model generated and saved to {args.output}")
        else:
            print(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())