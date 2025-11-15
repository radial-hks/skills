# LangChain Agent Creation Patterns

## Tool Creation Patterns

### Basic Tool Structure

To create a basic LangChain tool:

```python
from langchain.tools import tool

@tool
def tool_name(param: str) -> str:
    """Tool description for the agent."""
    return "result"
```

### Advanced Tool with Schema Validation

To create tools with input validation:

```python
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

class ToolInput(BaseModel):
    param: str = Field(description="Parameter description")
    optional_param: int = Field(default=0, description="Optional parameter")

def tool_function(param: str, optional_param: int = 0) -> str:
    return f"Processed: {param}"

tool = StructuredTool.from_function(
    func=tool_function,
    name="tool_name",
    description="Tool description",
    args_schema=ToolInput
)
```

### Error Handling Pattern

To implement robust error handling in tools:

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

## Agent Configuration Patterns

### Basic Agent Creation

To create a basic agent:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-3-sonnet-20240229",
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant"
)
```

### Multi-Tool Agent

To create agents with multiple specialized tools:

```python
agent = create_agent(
    model="claude-3-sonnet-20240229",
    tools=[search_tool, calculator, data_processor],
    system_prompt="You are a helpful assistant with search, calculation, and data processing capabilities"
)
```

### Custom System Prompt

To create domain-specific agents:

```python
system_prompt = """You are a data analysis assistant. When given data:
1. Always validate the data format first
2. Provide statistical summaries
3. Suggest appropriate visualizations
4. Identify potential insights
5. Flag any data quality issues

Be concise and focus on actionable insights."""

agent = create_agent(
    model="claude-3-sonnet-20240229",
    tools=[analyze_data, create_chart],
    system_prompt=system_prompt
)
```

## Testing Patterns

### Unit Testing Tools

To test individual tools:

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
```

### Integration Testing Agents

To test complete agent workflows:

```python
def test_agent_workflow():
    """Test complete agent workflow."""
    # Create test tools
    @tool
    def mock_tool(query: str) -> str:
        return f"Mock result for: {query}"
    
    # Create agent
    agent = create_agent(
        model="claude-3-haiku-20240307",
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
```

## Performance Optimization

### Model Selection

To choose appropriate models:

- **Simple tasks**: Use `claude-3-haiku-20240307` for cost efficiency
- **Complex reasoning**: Use `claude-3-sonnet-20240229` for balanced performance
- **Advanced analysis**: Use `claude-3-opus-20240229` for maximum capability

### Tool Optimization

To optimize tool performance:

1. **Cache repeated operations**
2. **Batch process multiple requests**
3. **Implement connection pooling**
4. **Use async tools for I/O operations**

### Response Streaming

To implement streaming for better user experience:

```python
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "message"}]
}):
    print(chunk.content, end="", flush=True)
```

## Error Handling Best Practices

### Common Error Types

| Error Type | Cause | Solution |
|------------|--------|----------|
| `ToolNotFound` | Tool name not recognized | Check tool registration and names |
| `InvalidInput` | Invalid input format | Validate input before agent call |
| `ModelError` | Model API issues | Check API keys and model availability |
| `TimeoutError` | Request timeout | Increase timeout or optimize tools |

### Error Recovery Strategies

To implement error recovery:

```python
def retry_on_error(max_retries: int = 3):
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
                    time.sleep(1 * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator
```

## Security Considerations

### Input Validation

To validate user inputs:

```python
import re

@tool
def validate_email(email: str) -> str:
    """Validate and process email addresses."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        return f"Error: Invalid email format - {email}"
    
    return f"Valid email: {email.lower()}"
```

### API Key Management

To secure API keys:

```python
import os

def get_model_config():
    """Get model configuration from environment."""
    return {
        "model_name": os.getenv("AGENT_MODEL", "claude-3-sonnet-20240229"),
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "temperature": float(os.getenv("AGENT_TEMPERATURE", "0.7"))
    }
```

## Deployment Patterns

### Environment Configuration

To configure agents for different environments:

```python
def create_production_agent(tools: list) -> Any:
    """Create agent with production configuration."""
    config = get_model_config()
    
    agent = create_agent(
        model=config["model_name"],
        tools=tools,
        system_prompt=get_production_prompt(),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 1000)
    )
    
    return agent
```

### Monitoring and Logging

To implement monitoring:

```python
import logging

class MonitoredAgent:
    def __init__(self, agent, logger_name: str = "agent"):
        self.agent = agent
        self.logger = logging.getLogger(logger_name)
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke agent with monitoring."""
        self.logger.info(f"Agent invoked with input: {input_data}")
        
        try:
            result = self.agent.invoke(input_data)
            self.logger.info("Agent completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Agent failed with error: {str(e)}")
            raise
```

## Multi-Agent Systems

### Agent Coordination

To create coordinated multi-agent systems:

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
            )
        }
    
    def route_query(self, query: str) -> str:
        """Route query to appropriate specialist agent."""
        if any(word in query.lower() for word in ["code", "function", "bug"]):
            return "code"
        elif any(word in query.lower() for word in ["data", "chart", "analysis"]):
            return "data"
        else:
            return "general"
```