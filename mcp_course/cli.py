"""Command-line interface for MCP Course."""

import json
from pathlib import Path

import typer

from mcp_course.content import CourseNavigator, FileSystemContentLoader
from mcp_course.utils.logging import setup_logging


app = typer.Typer(
    name="mcp-course",
    help="Interactive course for learning Model Context Protocol with Ollama",
    no_args_is_help=True
)


@app.command()
def setup(
    content_dir: Path | None = typer.Option(
        None,
        "--content-dir",
        "-c",
        help="Directory containing course content"
    )
) -> None:
    """Set up the MCP course environment."""
    setup_logging()

    if content_dir is None:
        content_dir = Path.cwd() / "content"

    typer.echo("🚀 Setting up MCP Course")
    typer.echo(f"📁 Content directory: {content_dir}")

    # Create content directory if it doesn't exist
    content_dir.mkdir(parents=True, exist_ok=True)

    # Create example module structure
    example_module_dir = content_dir / "01-mcp-fundamentals"
    example_module_dir.mkdir(exist_ok=True)

    # Create example module.json
    module_config = {
        "title": "MCP Fundamentals",
        "description": "Introduction to Model Context Protocol concepts and architecture",
        "order": 1,
        "difficulty_level": "beginner",
        "learning_objectives": [
            {
                "id": "obj_001",
                "description": "Understand MCP architecture and components",
                "level": "beginner"
            }
        ]
    }


    with Path.open(example_module_dir / "module.json", "w") as f:
        json.dump(module_config, f, indent=2)

    # Create lessons directory and example lesson
    lessons_dir = example_module_dir / "lessons"
    lessons_dir.mkdir(exist_ok=True)

    lesson_config = {
        "title": "What is MCP?",
        "description": "Learn the basics of Model Context Protocol",
        "order": 1,
        "exercises": [
            {
                "id": "ex_001",
                "title": "MCP Concepts Quiz",
                "description": "Test your understanding of MCP basics",
                "difficulty": "easy",
                "estimated_minutes": 10
            }
        ]
    }

    with Path.open(lessons_dir / "lesson_001.json", "w") as f:
        json.dump(lesson_config, f, indent=2)

    typer.echo("✅ Course structure created successfully!")
    typer.echo(f"📚 Example module created at: {example_module_dir}")
    typer.echo("🎯 You can now start adding your course content.")


@app.command()
def list_modules(
    content_dir: Path | None = typer.Option(
        None,
        "--content-dir",
        "-c",
        help="Directory containing course content"
    )
) -> None:
    """List available course modules."""
    setup_logging()

    if content_dir is None:
        content_dir = Path.cwd() / "content"

    if not content_dir.exists():
        typer.echo(f"❌ Content directory not found: {content_dir}")
        typer.echo("💡 Run 'mcp-course setup' to create the course structure.")
        raise typer.Exit(1)

    try:
        loader = FileSystemContentLoader(content_dir)
        modules = loader.get_available_modules()

        if not modules:
            typer.echo("📭 No modules found in the content directory.")
            typer.echo("💡 Run 'mcp-course setup' to create example content.")
            return

        typer.echo(f"📚 Found {len(modules)} module(s):")

        for module_id in modules:
            module = loader.load_module(module_id)
            if module:
                typer.echo(f"  • {module.title} ({module_id})")
                typer.echo(f"    📖 {len(module.lessons)} lesson(s)")
                typer.echo(f"    ⏱️  ~{module.estimated_duration_minutes} minutes")
                typer.echo(f"    🎯 {module.difficulty_level}")
                typer.echo()
            else:
                typer.echo(f"  • {module_id} (failed to load)")

    except Exception as e:
        typer.echo(f"❌ Error loading modules: {e}")
        raise typer.Exit(1) from e


@app.command()
def navigate(
    content_dir: Path | None = typer.Option(
        None,
        "--content-dir",
        "-c",
        help="Directory containing course content"
    )
) -> None:
    """Navigate through course content."""
    setup_logging()

    if content_dir is None:
        content_dir = Path.cwd() / "content"

    if not content_dir.exists():
        typer.echo(f"❌ Content directory not found: {content_dir}")
        typer.echo("💡 Run 'mcp-course setup' to create the course structure.")
        raise typer.Exit(1)

    try:
        loader = FileSystemContentLoader(content_dir)
        navigator = CourseNavigator(loader)

        context = navigator.get_navigation_context()
        summary = navigator.get_course_summary(context)

        typer.echo("🧭 Course Navigation")
        typer.echo("=" * 40)
        typer.echo(f"📚 Total modules: {summary['total_modules']}")
        typer.echo(f"📖 Total lessons: {summary['total_lessons']}")

        if context.current_module_id:
            typer.echo(f"📍 Current module: {context.current_module_id}")
            if context.current_lesson_id:
                typer.echo(f"📍 Current lesson: {context.current_lesson_id}")

        typer.echo("\n🗺️  Learning Path:")
        path = navigator.get_learning_path(context)
        for i, (module_id, lesson_id) in enumerate(path, 1):
            typer.echo(f"  {i}. {module_id} → {lesson_id}")

    except Exception as e:
        typer.echo(f"❌ Error navigating course: {e}")
        raise typer.Exit(1) from e


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo("MCP Course v0.1.0")
    typer.echo("Interactive course for learning Model Context Protocol with Ollama")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
