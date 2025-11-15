---
name: langchain-python-agent-creator
description: Create and configure LangChain Python agents with tools, prompts, and models. Use when building LLM-powered agents, chatbots, or AI applications with tool-calling capabilities.
---

# LangChain Python Agent Creator

## Purpose

To create LangChain Python agents with proper architecture, tool integration, and best practices. This skill provides the procedural knowledge needed to build LLM-powered agents, chatbots, and AI applications with tool-calling capabilities using LangChain's agent framework.

## When to Use This Skill

Use this skill when users need to:
- Create LangChain agents with custom tools and prompts
- Build LLM-powered applications with tool-calling capabilities
- Implement multi-step workflows with AI agents
- Develop chatbots with external tool integration
- Design AI applications that require deterministic tool execution

## How to Use This Skill

To create LangChain Python agents, follow these steps:

1. **Start with basic agent creation** using the quick start examples
2. **Define custom tools** using the tool creation patterns
3. **Configure appropriate models** based on complexity requirements
4. **Implement error handling** using the provided patterns
5. **Test agent functionality** with the testing utilities
6. **Apply specialized templates** for common use cases

### Quick Start Implementation

To create a basic LangChain agent:

1. Import required modules:
```python
from langchain.agents import create_agent
from langchain.tools import tool
```

2. Define tools with proper descriptions:
```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
```

3. Create the agent:
```python
agent = create_agent(
    model="claude-3-sonnet-20240229",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
```

4. Execute the agent:
```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather in sf"}]
})
```

### Advanced Implementation Patterns

To implement advanced agent patterns:

1. **Use multi-tool agents** for complex workflows:
   - Define multiple specialized tools
   - Combine tools that work well together
   - Implement proper error handling for each tool

2. **Implement custom system prompts**:
   - Create domain-specific instructions
   - Define clear agent behavior guidelines
   - Include tool usage instructions

3. **Apply testing frameworks**:
   - Use the provided testing utilities to validate agent functionality
   - Implement performance testing for optimization
   - Test error handling and edge cases

### Using Bundled Resources

To leverage the bundled utilities:

1. **Use scripts/agent_utils.py** for common patterns:
   - Tool validation and creation
   - Error handling wrappers
   - Multi-tool agent configurations
   - See references/agent_patterns.md for detailed usage

2. **Apply scripts/agent_templates.py** for pre-built solutions:
   - Customer support agents
   - Data analysis agents
   - Code review agents
   - Content writing agents
   - Research assistant agents

3. **Implement testing with scripts/test_agents.py**:
   - Performance benchmarking
   - Error handling validation
   - Tool usage analysis
   - Comprehensive test reporting

4. **Reference detailed documentation**:
   - references/agent_patterns.md: Tool creation and agent configuration patterns
   - references/model_integration.md: Model selection and configuration guide

### Best Practices Implementation

To ensure high-quality agent creation:

1. **Follow tool design principles**:
   - Write clear, comprehensive tool descriptions
   - Use specific parameter names and types
   - Implement robust error handling
   - Return structured, useful information

2. **Configure agents appropriately**:
   - Choose models based on complexity and cost requirements
   - Keep system prompts concise and specific
   - Include only relevant tools

3. **Test thoroughly**:
   - Test individual tools in isolation
   - Validate complete agent workflows
   - Handle edge cases and unusual inputs

### Common Use Case Implementations

To implement specific agent types:

1. **Customer Support Agent**:
   - Use knowledge base search tools
   - Implement ticket creation capabilities
   - Include escalation mechanisms

2. **Data Analysis Agent**:
   - Load and validate datasets
   - Generate statistical summaries
   - Create appropriate visualizations

3. **Code Review Agent**:
   - Analyze code quality and structure
   - Check for security vulnerabilities
   - Suggest performance improvements

### Integration Patterns

To integrate agents with existing systems:

1. **API Integration**:
   - Wrap existing APIs as tools
   - Maintain consistent error handling
   - Use appropriate authentication

2. **Web Framework Integration**:
   - Create RESTful endpoints
   - Handle concurrent requests
   - Implement proper error responses

### Troubleshooting Implementation Issues

To resolve common problems:

1. **Tool not being called**: Check tool descriptions and parameter names
2. **Agent giving wrong answers**: Improve system prompt and tool return formats
3. **Performance issues**: Consider simpler models or optimize tool implementations

For detailed API reference and advanced patterns, see the reference materials in this skill.