# Frequently Asked Questions

## General Questions

## Protocol Questions

### What's the difference between MCP tools and resources?

Tools are functions that can be called by LLMs to perform actions or computations. They take parameters and return results. Resources are data sources that provide information to LLMs, like files, databases, or APIs. Tools are active (they do something), while resources are passive (they provide data).

**Example:**
```python
# Tool example - performs an action
def calculate_tool(params):
    return {"result": params["a"] + params["b"]}

# Resource example - provides data  
def get_document_resource(uri):
    with open(uri, 'r') as f:
        return {"content": f.read()}
```

### How do I debug MCP protocol messages?

Enable logging in both client and server to see all protocol messages. Use the logging configuration to capture JSON-RPC messages, and check for proper message formatting, required fields, and response handling.

**Example:**
```python
import logging

# Enable MCP protocol logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('mcp')

# In your client/server code
logger.debug(f"Sending message: {message}")
logger.debug(f"Received response: {response}")
```

## Ollama Questions

### Why is Ollama using so much memory?

Ollama loads entire models into memory for fast inference. Large models (7B+ parameters) can use 4-16GB+ RAM. Use smaller models for development, adjust Ollama's memory settings, or use model quantization to reduce memory usage.

**Example:**
```python
# Check model sizes before loading
import requests

def get_model_info():
    response = requests.get("http://localhost:11434/api/tags")
    models = response.json()
    
    for model in models['models']:
        size_gb = model['size'] / (1024**3)
        print(f"{model['name']}: {size_gb:.1f} GB")
```

## Tools Questions

### How do I handle async operations in MCP servers?

MCP servers should use async/await patterns for I/O operations. All tool and resource handlers should be async functions. Use proper error handling and timeouts for external API calls.

**Example:**
```python
import asyncio
from mcp.server import Server

app = Server("my-server")

@app.tool()
async def async_tool(params):
    # Use async for I/O operations
    async with aiohttp.ClientSession() as session:
        async with session.get(params['url']) as response:
            return await response.text()

# Run server with asyncio
if __name__ == "__main__":
    asyncio.run(app.run())
```

## Performance Questions

### What should I do if tool execution is too slow?

Implement timeouts, use async operations, cache results when possible, and consider breaking large operations into smaller chunks. For long-running tasks, consider returning progress updates or using background processing.

**Example:**
```python
import asyncio
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_computation(input_data):
    # Expensive computation here
    return result

async def optimized_tool(params):
    try:
        # Set timeout for operations
        result = await asyncio.wait_for(
            slow_operation(params),
            timeout=30.0
        )
        return result
    except asyncio.TimeoutError:
        return {"error": "Operation timed out"}
```

