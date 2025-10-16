# MCP Course Troubleshooting Guide

This guide provides solutions for common issues encountered when learning and implementing MCP (Model Context Protocol) systems with Ollama.

## Quick Reference

### Common Issues by Category

#### Connection Issues
- [MCP Server Connection Refused](#mcp-server-connection-refused)

#### Ollama Issues
- [Ollama Model Not Loading](#ollama-model-not-loading)

#### Tools Issues
- [Tool Call Validation Errors](#tool-call-validation-errors)

#### Resources Issues
- [Resource Access Permissions](#resource-access-permissions)

## Detailed Solutions

### MCP Server Connection Refused

**Category:** connection

**Symptoms:**
- Client cannot connect to MCP server
- Connection timeout errors
- Server not responding to initialize requests

**Possible Causes:**
- Server not started or crashed
- Incorrect server configuration
- Port conflicts or firewall issues
- Transport protocol mismatch

**Solutions:**
- Verify server is running with proper logging
- Check server configuration and port settings
- Ensure client and server use same transport protocol
- Review firewall and network settings

**Code Examples:**

Example 1:
```python
# Check if server is running
import subprocess
import sys

def check_server_process(server_name):
    try:
        result = subprocess.run(['pgrep', '-f', server_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Server {server_name} is running (PID: {result.stdout.strip()})")
        else:
            print(f"Server {server_name} is not running")
    except Exception as e:
        print(f"Error checking server: {e}")
```

Example 2:
```python
# Test basic server connection
import asyncio
from mcp import ClientSession, StdioServerParameters

async def test_connection():
    try:
        server_params = StdioServerParameters(
            command="python", 
            args=["your_server.py"]
        )
        
        async with ClientSession(server_params) as session:
            await session.initialize()
            print("Connection successful!")
            
    except Exception as e:
        print(f"Connection failed: {e}")

asyncio.run(test_connection())
```

### Ollama Model Not Loading

**Category:** ollama

**Symptoms:**
- Ollama returns 'model not found' errors
- Model loading takes extremely long time
- Out of memory errors during model loading

**Possible Causes:**
- Model not downloaded or installed
- Insufficient system memory
- Corrupted model files
- Ollama service not running

**Solutions:**
- Download model using 'ollama pull <model_name>'
- Check available system memory and close other applications
- Reinstall model if corrupted
- Start Ollama service and verify it's running

**Code Examples:**

Example 1:
```python
# Check Ollama service status
import requests
import json

def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("Available models:")
            for model in models.get('models', []):
                print(f"  - {model['name']}")
        else:
            print("Ollama service not responding")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
```

Example 2:
```python
# Download and verify model
import subprocess

def setup_ollama_model(model_name="llama2"):
    try:
        # Pull the model
        print(f"Downloading {model_name}...")
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Model {model_name} downloaded successfully")
            
            # Test the model
            test_result = subprocess.run(['ollama', 'run', model_name, 'Hello'], 
                                       capture_output=True, text=True, timeout=30)
            if test_result.returncode == 0:
                print("Model is working correctly")
            else:
                print("Model download succeeded but testing failed")
        else:
            print(f"Failed to download model: {result.stderr}")
            
    except Exception as e:
        print(f"Error setting up model: {e}")
```

### Tool Call Validation Errors

**Category:** tools

**Symptoms:**
- Tool parameters rejected by server
- JSON schema validation failures
- Tool execution returns unexpected errors

**Possible Causes:**
- Incorrect parameter types or formats
- Missing required parameters
- Tool schema definition errors
- Parameter validation logic bugs

**Solutions:**
- Validate tool parameters against schema before calling
- Check tool schema definition for accuracy
- Add comprehensive parameter validation in tool implementation
- Use proper type conversion and error handling

**Code Examples:**

Example 1:
```python
# Validate tool parameters before calling
import jsonschema
from jsonschema import validate

def validate_tool_params(tool_schema, params):
    try:
        validate(instance=params, schema=tool_schema['inputSchema'])
        return True, None
    except jsonschema.exceptions.ValidationError as e:
        return False, str(e)

# Example usage
tool_schema = {
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1}
        },
        "required": ["query"]
    }
}

params = {"query": "test", "limit": 10}
is_valid, error = validate_tool_params(tool_schema, params)
if not is_valid:
    print(f"Validation error: {error}")
```

Example 2:
```python
# Robust tool implementation with validation
from typing import Any, Dict

class MCPTool:
    def __init__(self, name: str, schema: Dict[str, Any]):
        self.name = name
        self.schema = schema
    
    def validate_and_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Validate parameters
            is_valid, error = validate_tool_params(self.schema, params)
            if not is_valid:
                return {
                    "error": f"Parameter validation failed: {error}",
                    "success": False
                }
            
            # Execute tool logic
            result = self._execute_tool(params)
            return {
                "result": result,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Tool execution failed: {str(e)}",
                "success": False
            }
    
    def _execute_tool(self, params: Dict[str, Any]) -> Any:
        # Implement actual tool logic here
        pass
```

### Resource Access Permissions

**Category:** resources

**Symptoms:**
- Permission denied when accessing resources
- Resource not found errors for existing files
- Inconsistent resource availability

**Possible Causes:**
- Insufficient file system permissions
- Incorrect resource URI formatting
- Security restrictions blocking access
- Resource path resolution issues

**Solutions:**
- Check and adjust file permissions
- Verify resource URI format and encoding
- Configure security settings appropriately
- Use absolute paths and proper URI schemes

**Code Examples:**

Example 1:
```python
# Check resource accessibility
import os
import urllib.parse

def check_resource_access(uri: str) -> Dict[str, Any]:
    try:
        # Parse URI
        parsed = urllib.parse.urlparse(uri)
        
        if parsed.scheme == 'file':
            file_path = parsed.path
            
            # Check if file exists
            if not os.path.exists(file_path):
                return {"accessible": False, "error": "File does not exist"}
            
            # Check read permissions
            if not os.access(file_path, os.R_OK):
                return {"accessible": False, "error": "No read permission"}
            
            # Get file info
            stat = os.stat(file_path)
            return {
                "accessible": True,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "permissions": oct(stat.st_mode)[-3:]
            }
        else:
            return {"accessible": False, "error": f"Unsupported URI scheme: {parsed.scheme}"}
            
    except Exception as e:
        return {"accessible": False, "error": str(e)}
```

Example 2:
```python
# Safe resource reader with proper error handling
import mimetypes
from pathlib import Path

class ResourceManager:
    def __init__(self, allowed_paths: List[str] = None):
        self.allowed_paths = allowed_paths or []
    
    def read_resource(self, uri: str) -> Dict[str, Any]:
        try:
            parsed = urllib.parse.urlparse(uri)
            
            if parsed.scheme != 'file':
                raise ValueError(f"Unsupported scheme: {parsed.scheme}")
            
            file_path = Path(parsed.path).resolve()
            
            # Security check: ensure path is allowed
            if self.allowed_paths:
                allowed = any(
                    str(file_path).startswith(str(Path(allowed).resolve()))
                    for allowed in self.allowed_paths
                )
                if not allowed:
                    raise PermissionError("Access to this path is not allowed")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            return {
                "content": content,
                "mimeType": mime_type or "text/plain",
                "size": len(content),
                "uri": uri
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to read resource {uri}: {str(e)}")
```

