# LangChain Python Agent Creator - Reference Guide

## API Reference

### Core Functions

#### `create_agent()`

Creates a LangChain agent with specified model, tools, and configuration.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-3-sonnet-20240229",  # Model name or LLM instance
    tools=[tool1, tool2],              # List of tools
    system_prompt="System prompt",     # Optional system prompt
    **kwargs                            # Additional configuration
)
```

**Parameters:**
- `model` (str | BaseChatModel): Model name (e.g., "claude-3-sonnet-20240229") or LLM instance
- `tools` (List[BaseTool]): List of tools available to the agent
- `system_prompt` (str, optional): System prompt for the agent
- `**kwargs`: Additional configuration options

**Returns:**
- `AgentExecutor`: Configured agent instance

#### `tool` Decorator

Converts a function into a LangChain tool.

```python
from langchain.tools import tool

@tool
def function_name(param: type) -> str:
    """Tool description for the agent."""
    return "result"
```

**Parameters:**
- Function with proper docstring and type hints

**Returns:**
- `BaseTool`: Tool instance that can be used by agents

### Model Integration

#### Claude Models

```python
# Direct model name
agent = create_agent(model="claude-3-sonnet-20240229", tools=[...])

# Anthropic Chat model
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
agent = create_agent(model=llm, tools=[...])
```

#### OpenAI Models

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent = create_agent(model=llm, tools=[...])
```

### Agent Execution

#### `invoke()` Method

Executes the agent with a given input.

```python
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "user message"}
    ]
})
```

**Parameters:**
- `input` (dict): Input dictionary with "messages" key containing conversation history

**Returns:**
- `dict`: Agent response with generated message and metadata

#### Streaming Responses

```python
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "message"}]
}):
    print(chunk.content, end="", flush=True)
```

## Tool Development

### Basic Tool Structure

```python
from langchain.tools import tool

@tool
def tool_name(param1: str, param2: int = 0) -> str:
    """
    Tool description that explains what it does.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter (optional)
    
    Returns:
        Description of return value
    """
    # Implementation
    return "result"
```

### Advanced Tool Features

#### Schema Definition

```python
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

class ToolInput(BaseModel):
    param1: str = Field(description="Description of param1")
    param2: int = Field(default=0, description="Description of param2")

def tool_function(param1: str, param2: int = 0) -> str:
    return "result"

tool = StructuredTool.from_function(
    func=tool_function,
    name="tool_name",
    description="Tool description",
    args_schema=ToolInput
)
```

#### Async Tools

```python
import asyncio
from langchain.tools import tool

@tool
async def async_tool(param: str) -> str:
    """Async tool description."""
    await asyncio.sleep(1)
    return f"Processed {param}"
```

### Error Handling Patterns

#### Graceful Error Handling

```python
@tool
def robust_tool(input_data: str) -> str:
    """Tool with comprehensive error handling."""
    try:
        # Validate input
        if not input_data or len(input_data.strip()) == 0:
            return "Error: Input cannot be empty"
        
        # Process data
        result = process_data(input_data)
        
        if result is None:
            return "Error: Processing failed - no result generated"
        
        return f"Success: {result}"
        
    except ValueError as e:
        return f"Error: Invalid input - {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error - {str(e)}"
```

#### Input Validation

```python
import re
from typing import Optional

@tool
def validate_email(email: str) -> str:
    """Validate and process email addresses."""
    # Email validation pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        return f"Error: Invalid email format - {email}"
    
    # Process valid email
    return f"Valid email: {email.lower()}"
```

## Agent Configuration Patterns

### System Prompt Templates

#### Role-Based Prompts

```python
def create_role_agent(role: str, expertise: list) -> str:
    """Create system prompt for role-based agents."""
    expertise_str = ", ".join(expertise)
    
    return f"""You are a {role} with expertise in {expertise_str}.

Your responsibilities:
1. Provide accurate and helpful information
2. Ask clarifying questions when needed
3. Suggest relevant tools and approaches
4. Maintain professional communication
5. Acknowledge limitations when appropriate

Always be concise and solution-oriented."""

# Usage
system_prompt = create_role_agent("data analyst", ["statistics", "visualization", "Python"])
agent = create_agent(model="claude-3-sonnet-20240229", tools=[...], system_prompt=system_prompt)
```

#### Task-Specific Prompts

```python
def create_task_prompt(task_type: str, constraints: list = None) -> str:
    """Create system prompt for specific tasks."""
    base_prompt = f"You are designed to {task_type}."
    
    if constraints:
        constraints_str = "\n".join([f"- {c}" for c in constraints])
        base_prompt += f"\n\nConstraints:\n{constraints_str}"
    
    return base_prompt

# Usage for code review
code_review_prompt = create_task_prompt(
    "review code for quality, security, and best practices",
    ["Focus on Python code", "Check for security vulnerabilities", "Suggest performance improvements"]
)
```

### Multi-Agent Systems

#### Agent Coordination

```python
class MultiAgentSystem:
    def __init__(self):
        self.specialists = {
            "code": create_agent(
                model="claude-3-sonnet-20240229",
                tools=[analyze_code, suggest_improvements],
                system_prompt="You are a code specialist."
            ),
            "data": create_agent(
                model="claude-3-sonnet-20240229",
                tools=[analyze_data, create_visualization],
                system_prompt="You are a data analysis specialist."
            ),
            "general": create_agent(
                model="claude-3-sonnet-20240229",
                tools=[search_web, calculate],
                system_prompt="You are a general purpose assistant."
            )
        }
    
    def route_query(self, query: str) -> str:
        """Route query to appropriate specialist agent."""
        # Simple routing based on keywords
        if any(word in query.lower() for word in ["code", "function", "bug"]):
            return "code"
        elif any(word in query.lower() for word in ["data", "chart", "analysis"]):
            return "data"
        else:
            return "general"
    
    def process_query(self, query: str) -> dict:
        """Process query with appropriate agent."""
        specialist = self.route_query(query)
        agent = self.specialists[specialist]
        
        return agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
```

## Testing and Validation

### Unit Testing Tools

```python
import pytest
from langchain.tools import tool

@tool
def test_tool(input_data: str) -> str:
    """Test tool for unit testing examples."""
    return f"Processed: {input_data}"

def test_tool_basic():
    """Test basic tool functionality."""
    result = test_tool.invoke({"input_data": "test"})
    assert result == "Processed: test"

def test_tool_empty_input():
    """Test tool with empty input."""
    result = test_tool.invoke({"input_data": ""})
    assert result == "Processed: "
```

### Integration Testing Agents

```python
def test_agent_workflow():
    """Test complete agent workflow."""
    # Create test tools
    @tool
    def mock_tool(query: str) -> str:
        return f"Mock result for: {query}"
    
    # Create agent
    agent = create_agent(
        model="claude-3-haiku-20240307",  # Use cheaper model for testing
        tools=[mock_tool],
        system_prompt="You are a test assistant."
    )
    
    # Test agent execution
    result = agent.invoke({
        "messages": [{"role": "user", "content": "test query"}]
    })
    
    # Validate response structure
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert "content" in result["messages"][-1]
```

### Performance Testing

```python
import time
from typing import List

def benchmark_agent(agent, test_queries: List[str], iterations: int = 10):
    """Benchmark agent performance."""
    times = []
    
    for query in test_queries:
        for _ in range(iterations):
            start_time = time.time()
            
            result = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    return {
        "average_time": avg_time,
        "max_time": max_time,
        "min_time": min_time,
        "total_queries": len(times)
    }
```

## Deployment Patterns

### Environment Configuration

```python
import os
from typing import Optional

def get_model_config() -> dict:
    """Get model configuration from environment."""
    return {
        "model_name": os.getenv("AGENT_MODEL", "claude-3-sonnet-20240229"),
        "temperature": float(os.getenv("AGENT_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("AGENT_MAX_TOKENS", "1000"))
    }

def create_production_agent(tools: list) -> str:
    """Create agent with production configuration."""
    config = get_model_config()
    
    agent = create_agent(
        model=config["model_name"],
        tools=tools,
        system_prompt=get_production_prompt(),
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )
    
    return agent
```

### Monitoring and Logging

```python
import logging
from typing import Dict, Any

class MonitoredAgent:
    def __init__(self, agent, logger_name: str = "agent"):
        self.agent = agent
        self.logger = logging.getLogger(logger_name)
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke agent with monitoring."""
        self.logger.info(f"Agent invoked with input: {input_data}")
        
        try:
            result = self.agent.invoke(input_data)
            self.logger.info(f"Agent completed successfully")
            return result
        
        except Exception as e:
            self.logger.error(f"Agent failed with error: {str(e)}")
            raise
    
    def stream(self, input_data: Dict[str, Any]):
        """Stream agent response with monitoring."""
        self.logger.info(f"Agent streaming started")
        
        try:
            for chunk in self.agent.stream(input_data):
                yield chunk
            self.logger.info(f"Agent streaming completed")
        
        except Exception as e:
            self.logger.error(f"Agent streaming failed: {str(e)}")
            raise
```

## Error Handling Reference

### Common Error Types

| Error Type | Cause | Solution |
|------------|--------|----------|
| `ToolNotFound` | Tool name not recognized | Check tool registration and names |
| `InvalidInput` | Invalid input format | Validate input before agent call |
| `ModelError` | Model API issues | Check API keys and model availability |
| `TimeoutError` | Request timeout | Increase timeout or optimize tools |
| `RateLimitError` | API rate limits | Implement rate limiting and retries |

### Error Recovery Strategies

```python
from typing import Optional, Dict, Any
import time
from functools import wraps

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying agent operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            
            return None
        return wrapper
    return decorator

@retry_on_error(max_retries=3)
def robust_agent_invoke(agent, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Invoke agent with retry logic."""
    return agent.invoke(input_data)
```

This reference guide provides comprehensive information for developing LangChain Python agents with proper patterns, error handling, and best practices.