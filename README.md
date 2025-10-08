# MCP Course with Ollama

A comprehensive, hands-on educational platform for learning Model Context Protocol (MCP) with local LLM deployment using Ollama.

## Overview

This course teaches developers how to build and integrate MCP Servers and Clients while maintaining privacy through local LLM processing. Learn MCP fundamentals, implement practical examples, and build real-world applications.

## Features

- **Privacy-First**: All AI processing stays local using Ollama
- **Hands-On Learning**: Interactive code examples and exercises
- **Progressive Curriculum**: From basics to advanced MCP patterns
- **Real-World Projects**: Build functional MCP applications

## Quick Start

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) for local LLM hosting

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vincentporte/mcp-course-ollama.git
cd mcp-course-ollama
```

2. Set up the environment with uv:
```bash
uv sync
```

3. Install Ollama and pull a model:
```bash
# Install Ollama (see https://ollama.com/download)
ollama pull llama3.2:3b
```

4. Set up the course:
```bash
uv run mcp-course setup
```

### Usage

Start the course:
```bash
uv run mcp-course info
```

Configure Ollama settings:
```bash
uv run mcp-course config --model llama3.2:3b --temperature 0.7
```

Demo ollama_client script, run the comprehensive demo to see all features:
```bash
python mcp_course/examples/ollama_integration_demo.py
```

Run the complete fundamentals course:
```bash
import asyncio
from mcp_course.fundamentals.main import run_complete_fundamentals_course

# Run the complete course
asyncio.run(run_complete_fundamentals_course())
```

## Course Structure

1. **MCP Fundamentals** - Core concepts and architecture
2. **Ollama Setup** - Local LLM configuration
3. **MCP Server Development** - Build custom servers
4. **MCP Client Integration** - Connect clients to servers
5. **Advanced Patterns** - Complex integrations and best practices

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Privacy & Security

This course is designed with privacy-by-design principles:
- All LLM processing happens locally via Ollama
- No data is sent to external services
- Local storage for progress and preferences
- Optional telemetry with explicit opt-in
