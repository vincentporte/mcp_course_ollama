#!/usr/bin/env python3
"""
MCP Server Template Generator

This utility generates MCP Server template code that students can use
as a starting point for their own server implementations.

Run this example:
    python -m mcp_course.examples.server_template_generator
"""
from pathlib import Path
import sys

from mcp_course.server.scaffolding import create_basic_server_template


def generate_server_template(output_path: str = "my_mcp_server.py") -> None:
    """
    Generate a basic MCP Server template file.

    Args:
        output_path: Path where the template file should be created
    """
    print("=== MCP Server Template Generator ===")
    print("Generating a basic MCP Server template...")
    print()

    # Get the template code
    template_code = create_basic_server_template()

    # Write to file
    output_file = Path(output_path)
    output_file.write_text(template_code)

    print(f"✅ Template generated: {output_file.absolute()}")
    print()
    print("The generated template includes:")
    print("- Basic server initialization")
    print("- Stdio transport setup")
    print("- Logging configuration")
    print("- Placeholder for adding tools and resources")
    print()
    print("To use the template:")
    print(f"1. Edit {output_path} to add your tools and resources")
    print(f"2. Run: python {output_path}")
    print("3. Connect an MCP Client to test your server")
    print()

    # Show a preview of the template
    lines = template_code.split('\n')
    print("Template preview (first 20 lines):")
    print("-" * 50)
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:2d}: {line}")
    if len(lines) > 20: # noqa
        print(f"... ({len(lines) - 20} more lines)")
    print("-" * 50)


def main():
    """Main entry point for the template generator."""
    try:
        generate_server_template()
    except Exception as e:
        print(f"❌ Error generating template: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
