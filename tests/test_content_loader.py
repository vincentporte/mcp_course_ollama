"""Tests for content loading system."""

import json
from pathlib import Path
import tempfile

import pytest

from mcp_course.content import FileSystemContentLoader


def test_filesystem_content_loader():
    """Test FileSystemContentLoader."""
    # Create temporary content structure
    with tempfile.TemporaryDirectory() as temp_dir:
        content_root = Path(temp_dir)

        # Create module directory
        module_dir = content_root / "module_001"
        module_dir.mkdir()

        # Create module configuration
        module_config = {
            "title": "Test Module",
            "description": "A test module",
            "order": 1,
            "difficulty_level": "beginner",
            "learning_objectives": [
                {
                    "id": "obj_001",
                    "description": "Test objective",
                    "level": "beginner"
                }
            ]
        }

        with Path.open(module_dir / "module.json", "w") as f:
            json.dump(module_config, f)

        # Create lessons directory
        lessons_dir = module_dir / "lessons"
        lessons_dir.mkdir()

        # Create lesson configuration
        lesson_config = {
            "title": "Test Lesson",
            "description": "A test lesson",
            "order": 1,
            "exercises": [
                {
                    "id": "ex_001",
                    "title": "Test Exercise",
                    "description": "A test exercise",
                    "difficulty": "easy",
                    "estimated_minutes": 10
                }
            ]
        }

        with Path.open(lessons_dir / "lesson_001.json", "w") as f:
            json.dump(lesson_config, f)

        # Test loader
        loader = FileSystemContentLoader(content_root)

        # Test getting available modules
        modules = loader.get_available_modules()
        assert "module_001" in modules

        # Test loading module
        module = loader.load_module("module_001")
        assert module is not None
        assert module.title == "Test Module"
        assert len(module.lessons) == 1

        # Test loading specific lesson
        lesson = loader.load_lesson("module_001", "lesson_001")
        assert lesson is not None
        assert lesson.title == "Test Lesson"
        assert len(lesson.exercises) == 1

        # Test cache
        module2 = loader.load_module("module_001")  # Should use cache
        assert module2 is module

        # Test reload
        module3 = loader.reload_module("module_001")
        assert module3 is not module  # Should be new instance

        # Test clear cache
        loader.clear_cache()
        module4 = loader.load_module("module_001")
        assert module4 is not module  # Should be new instance


def test_loader_nonexistent_content():
    """Test loader with nonexistent content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        content_root = Path(temp_dir)
        loader = FileSystemContentLoader(content_root)

        # Test with no modules
        modules = loader.get_available_modules()
        assert len(modules) == 0

        # Test loading nonexistent module
        module = loader.load_module("nonexistent")
        assert module is None

        # Test loading nonexistent lesson
        lesson = loader.load_lesson("nonexistent", "nonexistent")
        assert lesson is None


def test_loader_invalid_content_root():
    """Test loader with invalid content root."""
    with pytest.raises(ValueError, match="Content root directory does not exist"):
        FileSystemContentLoader(Path("/nonexistent/path"))


def test_loader_module_without_lessons():
    """Test loading module without lessons directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        content_root = Path(temp_dir)

        # Create module directory without lessons
        module_dir = content_root / "module_001"
        module_dir.mkdir()

        module_config = {
            "title": "Test Module",
            "description": "A test module",
            "order": 1
        }

        with Path.open(module_dir / "module.json", "w") as f:
            json.dump(module_config, f)

        loader = FileSystemContentLoader(content_root)
        module = loader.load_module("module_001")

        assert module is not None
        assert module.title == "Test Module"
        assert len(module.lessons) == 0


def test_loader_lesson_from_directory():
    """Test loading lesson from directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        content_root = Path(temp_dir)

        # Create module and lesson directories
        module_dir = content_root / "module_001"
        module_dir.mkdir()

        module_config = {"title": "Test Module", "description": "Test", "order": 1}
        with Path.open(module_dir / "module.json", "w") as f:
            json.dump(module_config, f)

        lessons_dir = module_dir / "lessons"
        lessons_dir.mkdir()

        lesson_dir = lessons_dir / "lesson_001"
        lesson_dir.mkdir()

        lesson_config = {
            "title": "Test Lesson",
            "description": "A test lesson",
            "order": 1
        }

        with Path.open(lesson_dir / "lesson.json", "w") as f:
            json.dump(lesson_config, f)

        # Create content file
        with Path.open(lesson_dir / "content.md", "w") as f:
            f.write("# Test Lesson Content\n\nThis is test content.")

        loader = FileSystemContentLoader(content_root)
        module = loader.load_module("module_001")

        assert module is not None
        assert len(module.lessons) == 1

        lesson = module.lessons[0]
        assert lesson.title == "Test Lesson"
        assert lesson.content_path is not None
        assert lesson.content_path.name == "content.md"
