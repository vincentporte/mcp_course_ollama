# MCP Protocol API Reference

## Protocol Overview

**Version:** 2024-11-05

Model Context Protocol for extending LLM capabilities

### Supported Capabilities
- tools
- resources
- prompts
- logging

### Core Concepts
- **servers**: Provide tools, resources, and prompts to clients
- **clients**: Connect to servers and utilize their capabilities
- **tools**: Functions that can be called by LLMs
- **resources**: Data sources accessible to LLMs
- **prompts**: Reusable prompt templates with parameters

## Message Types

### initialize

**Description:** Initialize connection between client and server
**Direction:** client -> server

**Example:**
```json
{'jsonrpc': '2.0', 'id': '1', 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}}, 'clientInfo': {'name': 'example-client', 'version': '1.0.0'}}}
```

### tools/list

**Description:** List all available tools from server
**Direction:** client -> server

**Example:**
```json
{'jsonrpc': '2.0', 'id': '2', 'method': 'tools/list'}
```

### tools/call

**Description:** Call a specific tool with parameters
**Direction:** client -> server

**Example:**
```json
{'jsonrpc': '2.0', 'id': '3', 'method': 'tools/call', 'params': {'name': 'get_weather', 'arguments': {'location': 'San Francisco'}}}
```

### resources/list

**Description:** List all available resources from server
**Direction:** client -> server

### resources/read

**Description:** Read content from a specific resource
**Direction:** client -> server

**Example:**
```json
{'jsonrpc': '2.0', 'id': '4', 'method': 'resources/read', 'params': {'uri': 'file:///path/to/document.txt'}}
```

## Tool Schema Examples

### simple_tool

```json
{'name': 'get_current_time', 'description': 'Get the current time in a specified timezone', 'inputSchema': {'type': 'object', 'properties': {'timezone': {'type': 'string', 'description': "Timezone identifier (e.g., 'America/New_York')", 'default': 'UTC'}}}}
```

### complex_tool

```json
{'name': 'search_database', 'description': 'Search a database with filters and pagination', 'inputSchema': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'Search query string'}, 'filters': {'type': 'object', 'properties': {'category': {'type': 'string'}, 'date_range': {'type': 'object', 'properties': {'start': {'type': 'string', 'format': 'date'}, 'end': {'type': 'string', 'format': 'date'}}}}}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100, 'default': 10}, 'offset': {'type': 'integer', 'minimum': 0, 'default': 0}}, 'required': ['query']}}
```

## Error Codes

### -32700 - Parse error

**Description:** Invalid JSON was received
**Example:** Malformed JSON in request

### -32600 - Invalid Request

**Description:** The JSON sent is not a valid Request object
**Example:** Missing required 'method' field

### -32601 - Method not found

**Description:** The method does not exist / is not available
**Example:** Calling non-existent tool or method

### -32602 - Invalid params

**Description:** Invalid method parameter(s)
**Example:** Tool parameters don't match schema

### -32603 - Internal error

**Description:** Internal JSON-RPC error
**Example:** Server-side processing error

