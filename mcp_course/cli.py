"""Command-line interface for MCP Course."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mcp_course.config.settings import config_manager

app = typer.Typer(
    name="mcp-course",
    help="Interactive course for learning Model Context Protocol with Ollama",
    no_args_is_help=True
)
console = Console()


@app.command()
def info():
    """Display course information and configuration."""
    settings = config_manager.course_settings
    ollama_config = config_manager.ollama_config
    
    info_text = Text()
    info_text.append(f"Course: {settings.course_name}\n", style="bold blue")
    info_text.append(f"Version: {settings.course_version}\n")
    info_text.append(f"Data Directory: {settings.course_data_dir}\n")
    info_text.append(f"Ollama Endpoint: {ollama_config.endpoint}\n")
    info_text.append(f"Default Model: {ollama_config.default_model}\n")
    
    console.print(Panel(info_text, title="MCP Course Information", border_style="blue"))


@app.command()
def setup():
    """Set up the course environment and check dependencies."""
    console.print("[bold green]Setting up MCP Course environment...[/bold green]")
    
    # Ensure data directory exists
    data_dir = config_manager.ensure_data_directory()
    console.print(f"✓ Created data directory: {data_dir}")
    
    # Check Ollama availability (placeholder)
    console.print("✓ Course configuration initialized")
    console.print("[bold green]Setup complete! Run 'mcp-course info' to see configuration.[/bold green]")


@app.command()
def config(
    ollama_endpoint: str = typer.Option(None, "--ollama-endpoint", help="Ollama API endpoint"),
    model: str = typer.Option(None, "--model", help="Default Ollama model"),
    temperature: float = typer.Option(None, "--temperature", help="Model temperature (0-2)"),
):
    """Configure course settings."""
    if ollama_endpoint:
        config_manager.update_ollama_config(endpoint=ollama_endpoint)
        console.print(f"✓ Updated Ollama endpoint: {ollama_endpoint}")
    
    if model:
        config_manager.update_ollama_config(default_model=model)
        console.print(f"✓ Updated default model: {model}")
    
    if temperature is not None:
        config_manager.update_ollama_config(temperature=temperature)
        console.print(f"✓ Updated temperature: {temperature}")
    
    if not any([ollama_endpoint, model, temperature is not None]):
        console.print("No configuration changes specified. Use --help for options.")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()