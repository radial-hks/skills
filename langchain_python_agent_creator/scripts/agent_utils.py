#!/usr/bin/env python3
"""
Agent Creation Utilities

This script provides utility functions for creating and managing LangChain agents
with common patterns and best practices.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Callable
from langchain.agents import create_agent
from langchain.tools import tool, StructuredTool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentCreationError(Exception):
    """Custom exception for agent creation errors."""
    pass

def validate_tools(tools: List[BaseTool]) -> bool:
    """
    Validate that all tools have proper descriptions and are callable.
    
    Args:
        tools: List of tools to validate
        
    Returns:
        True if all tools are valid
        
    Raises:
        AgentCreationError: If tools are invalid
    """
    if not tools:
        raise AgentCreationError("At least one tool must be provided")
    
    for tool in tools:
        if not hasattr(tool, 'description') or not tool.description:
            raise AgentCreationError(f"Tool {tool.name} missing description")
        
        if not callable(tool):
            raise AgentCreationError(f"Tool {tool.name} is not callable")
    
    return True

def create_structured_tool(
    name: str,
    description: str,
    func: Callable,
    input_schema: BaseModel
) -> StructuredTool:
    """
    Create a structured tool with proper schema validation.
    
    Args:
        name: Tool name
        description: Tool description
        func: Function to execute
        input_schema: Pydantic model for input validation
        
    Returns:
        Configured StructuredTool instance
    """
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=input_schema
    )

def create_error_handling_tool(
    original_func: Callable,
    error_message: str = "Tool execution failed"
) -> Callable:
    """
    Wrap a tool function with error handling.
    
    Args:
        original_func: Original tool function
        error_message: Custom error message
        
    Returns:
        Wrapped function with error handling
    """
    def wrapped_func(*args, **kwargs):
        try:
            return original_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{error_message}: {str(e)}")
            return f"Error: {error_message} - {str(e)}"
    
    return wrapped_func

def create_validation_tool(
    validation_func: Callable[[Any], bool],
    error_message: str
) -> Callable:
    """
    Create a validation tool that checks input before processing.
    
    Args:
        validation_func: Function that returns True if input is valid
        error_message: Error message to return if validation fails
        
    Returns:
        Validation function
    """
    def validator(input_data: Any) -> str:
        if validation_func(input_data):
            return "Validation passed"
        else:
            return f"Validation failed: {error_message}"
    
    return validator

def create_agent_with_validation(
    model: str,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    validate_tools_flag: bool = True,
    **kwargs
) -> Any:
    """
    Create an agent with optional tool validation and error handling.
    
    Args:
        model: Model name or instance
        tools: List of tools
        system_prompt: Optional system prompt
        validate_tools_flag: Whether to validate tools before creation
        **kwargs: Additional arguments for create_agent
        
    Returns:
        Configured agent instance
        
    Raises:
        AgentCreationError: If agent creation fails
    """
    try:
        if validate_tools_flag:
            validate_tools(tools)
        
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            **kwargs
        )
        
        logger.info(f"Agent created successfully with {len(tools)} tools")
        return agent
        
    except Exception as e:
        raise AgentCreationError(f"Failed to create agent: {str(e)}")

def create_multi_tool_agent(
    model: str,
    tool_categories: Dict[str, List[BaseTool]],
    system_prompt: str,
    **kwargs
) -> Any:
    """
    Create an agent with tools organized by categories.
    
    Args:
        model: Model name or instance
        tool_categories: Dictionary mapping category names to tool lists
        system_prompt: System prompt describing agent purpose
        **kwargs: Additional arguments for create_agent
        
    Returns:
        Configured agent instance
    """
    all_tools = []
    category_info = []
    
    for category, tools in tool_categories.items():
        all_tools.extend(tools)
        category_info.append(f"{category}: {len(tools)} tools")
    
    # Enhance system prompt with tool category information
    enhanced_prompt = f"""{system_prompt}

Available tool categories:
{chr(10).join(category_info)}

Use appropriate tools based on the user's request."""
    
    return create_agent_with_validation(
        model=model,
        tools=all_tools,
        system_prompt=enhanced_prompt,
        **kwargs
    )

def create_sequential_tool_agent(
    model: str,
    tools: List[BaseTool],
    system_prompt: str,
    max_iterations: int = 10,
    **kwargs
) -> Any:
    """
    Create an agent optimized for sequential tool usage.
    
    Args:
        model: Model name or instance
        tools: List of tools that work well in sequence
        system_prompt: System prompt
        max_iterations: Maximum number of tool calls
        **kwargs: Additional arguments for create_agent
        
    Returns:
        Configured agent instance
    """
    sequential_prompt = f"""{system_prompt}

You have access to tools that work well in sequence. When solving problems:
1. Break down complex tasks into steps
2. Use tools sequentially when appropriate
3. Combine results from multiple tools
4. Provide comprehensive final answers

You can make up to {max_iterations} tool calls to solve a problem."""
    
    return create_agent_with_validation(
        model=model,
        tools=tools,
        system_prompt=sequential_prompt,
        **kwargs
    )

def save_agent_config(agent: Any, config_path: str) -> bool:
    """
    Save agent configuration to a file.
    
    Args:
        agent: Agent instance
        config_path: Path to save configuration
        
    Returns:
        True if successful
    """
    try:
        config = {
            "model": str(agent.model),
            "tools": [tool.name for tool in agent.tools],
            "tool_count": len(agent.tools)
        }
        
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        logger.info(f"Agent configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save agent config: {str(e)}")
        return False

def load_agent_config(config_path: str) -> Dict[str, Any]:
    """
    Load agent configuration from a file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
        
        logger.info(f"Agent configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load agent config: {str(e)}")
        return {}

# Example usage and testing functions
def create_example_tools() -> List[BaseTool]:
    """Create example tools for testing."""
    
    @tool
    def calculator(expression: str) -> str:
        """Evaluate mathematical expressions."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Error: Invalid expression"
    
    @tool
    def string_processor(text: str, operation: str = "upper") -> str:
        """Process strings with various operations."""
        operations = {
            "upper": text.upper(),
            "lower": text.lower(),
            "reverse": text[::-1],
            "length": str(len(text))
        }
        
        if operation in operations:
            return operations[operation]
        else:
            return f"Error: Unknown operation '{operation}'"
    
    @tool
    def data_validator(data: str) -> str:
        """Validate data format and content."""
        if not data:
            return "Error: Empty data"
        
        if len(data) < 3:
            return "Error: Data too short"
        
        return f"Valid data: {len(data)} characters"
    
    return [calculator, string_processor, data_validator]

def test_agent_creation():
    """Test agent creation with example tools."""
    try:
        # Create example tools
        tools = create_example_tools()
        
        # Create agent
        agent = create_agent_with_validation(
            model="claude-3-haiku-20240307",
            tools=tools,
            system_prompt="You are a helpful assistant with calculation and text processing capabilities."
        )
        
        # Test agent
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Calculate 2 + 2 and process the result as uppercase"}]
        })
        
        print("Agent creation test successful!")
        print(f"Response: {result}")
        
        return True
        
    except Exception as e:
        print(f"Agent creation test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run tests
    print("Testing agent creation utilities...")
    test_agent_creation()