# LangChain Model Integration Guide

## Supported Models

### Anthropic Claude Models

To integrate Claude models:

```python
# Direct model name
agent = create_agent(model="claude-3-sonnet-20240229", tools=[...])

# Anthropic Chat model with custom configuration
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.7,
    max_tokens=1000
)
agent = create_agent(model=llm, tools=[...])
```

**Available Claude Models:**
- `claude-3-haiku-20240307`: Fastest, most cost-effective for simple tasks
- `claude-3-sonnet-20240229`: Balanced performance for most use cases
- `claude-3-opus-20240229`: Most capable for complex reasoning tasks

### OpenAI Models

To integrate OpenAI models:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    openai_api_key="your-api-key"
)
agent = create_agent(model=llm, tools=[...])
```

**Available OpenAI Models:**
- `gpt-4`: Most capable GPT model
- `gpt-4-turbo`: Faster, more cost-effective GPT-4
- `gpt-3.5-turbo`: Fast and cost-effective for simpler tasks

### Other Model Providers

To integrate other model providers:

```python
# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro")
agent = create_agent(model=gemini_llm, tools=[...])

# Local models via Ollama
from langchain_community.llms import Ollama

ollama_llm = Ollama(model="llama2")
agent = create_agent(model=ollama_llm, tools=[...])
```

## Model Selection Guidelines

### Task Complexity Matrix

| Task Type | Recommended Model | Reasoning |
|-----------|------------------|-----------|
| Simple Q&A, basic calculations | claude-3-haiku-20240307 or gpt-3.5-turbo | Fast, cost-effective |
| Multi-step reasoning, analysis | claude-3-sonnet-20240229 or gpt-4-turbo | Balanced performance |
| Complex problem solving, coding | claude-3-opus-20240229 or gpt-4 | Maximum capability |
| Real-time applications | claude-3-haiku-20240307 | Fastest response times |
| Cost-sensitive applications | gpt-3.5-turbo or claude-3-haiku-20240307 | Most cost-effective |

### Performance Considerations

**Latency Comparison:**
- Claude Haiku: ~1-2 seconds
- Claude Sonnet: ~2-4 seconds  
- Claude Opus: ~4-8 seconds
- GPT-3.5-turbo: ~1-2 seconds
- GPT-4-turbo: ~2-4 seconds
- GPT-4: ~4-8 seconds

**Cost Comparison (per 1K tokens):**
- Claude Haiku: $0.00025 input / $0.00125 output
- Claude Sonnet: $0.003 input / $0.015 output
- Claude Opus: $0.015 input / $0.075 output
- GPT-3.5-turbo: $0.0005 input / $0.0015 output
- GPT-4-turbo: $0.01 input / $0.03 output
- GPT-4: $0.03 input / $0.06 output

## Configuration Parameters

### Temperature Settings

To control randomness in responses:

```python
# Conservative, predictable responses
agent = create_agent(
    model="claude-3-sonnet-20240229",
    tools=[...],
    temperature=0.1  # Low temperature
)

# Creative, varied responses  
agent = create_agent(
    model="claude-3-sonnet-20240229",
    tools=[...],
    temperature=0.8  # High temperature
)
```

**Temperature Guidelines:**
- `0.0-0.3`: Factual, deterministic tasks
- `0.4-0.6`: Balanced reasoning tasks
- `0.7-1.0`: Creative, generative tasks

### Token Limits

To control response length:

```python
agent = create_agent(
    model="claude-3-sonnet-20240229",
    tools=[...],
    max_tokens=500  # Limit response length
)
```

**Token Limit Recommendations:**
- Simple tasks: 100-300 tokens
- Complex analysis: 500-1000 tokens
- Code generation: 1000-2000 tokens
- Document creation: 2000+ tokens

## Environment Configuration

### API Key Management

To securely manage API keys:

```python
import os
from langchain_anthropic import ChatAnthropic

def create_secure_agent(tools: list) -> Any:
    """Create agent with secure API key management."""
    
    # Load from environment variables
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")
    
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        anthropic_api_key=anthropic_api_key
    )
    
    return create_agent(model=llm, tools=tools)
```

### Model Configuration

To configure models for different environments:

```python
def get_model_config(environment: str = "development") -> dict:
    """Get model configuration based on environment."""
    
    configs = {
        "development": {
            "model": "claude-3-haiku-20240307",
            "temperature": 0.7,
            "max_tokens": 500
        },
        "staging": {
            "model": "claude-3-sonnet-20240229", 
            "temperature": 0.5,
            "max_tokens": 1000
        },
        "production": {
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.3,
            "max_tokens": 1500
        }
    }
    
    return configs.get(environment, configs["development"])
```

## Fallback Strategies

### Model Fallback

To implement model fallback for reliability:

```python
def create_agent_with_fallback(primary_model: str, fallback_models: list, tools: list) -> Any:
    """Create agent with model fallback strategy."""
    
    models_to_try = [primary_model] + fallback_models
    
    for model in models_to_try:
        try:
            agent = create_agent(model=model, tools=tools)
            # Test the agent
            test_result = agent.invoke({
                "messages": [{"role": "user", "content": "test"}]
            })
            return agent
            
        except Exception as e:
            print(f"Model {model} failed: {str(e)}")
            continue
    
    raise RuntimeError("All models failed")

# Usage
agent = create_agent_with_fallback(
    primary_model="claude-3-sonnet-20240229",
    fallback_models=["claude-3-haiku-20240307", "gpt-4-turbo"],
    tools=[tool1, tool2]
)
```

### Rate Limiting

To handle rate limiting:

```python
import time
from functools import wraps

def rate_limit(calls_per_second: int = 1):
    """Decorator for rate limiting agent calls."""
    min_interval = 1.0 / calls_per_second
    
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        
        return wrapper
    return decorator

@rate_limit(calls_per_second=0.5)  # 1 call per 2 seconds
def invoke_agent(agent, input_data):
    return agent.invoke(input_data)
```