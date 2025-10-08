#!/usr/bin/env python3
"""
Test script to verify the course content management system implementation.
This script tests all the components we just implemented.
"""

import json
from pathlib import Path
import tempfile

from mcp_course.content import CourseNavigator, FileSystemContentLoader
from mcp_course.models import CourseModule, Exercise, LearningObjective, Lesson


# Test imports
def test_imports():
    """Test that all modules can be imported successfully."""

    # If we get here without ImportError, the test passes
    assert True


def test_content_models():
    """Test content structure models."""


    # Test LearningObjective
    objective = LearningObjective(
        id="obj_001",
        description="Understand MCP basics",
        level="beginner",
        skills=["mcp", "protocol"]
    )

    # Test Exercise
    exercise = Exercise(
        id="ex_001",
        title="Hello MCP",
        description="Create your first MCP server",
        difficulty="easy",
        estimated_minutes=15,
        code_template="# Your code here",
        hints=["Start with imports", "Define server class"]
    )

    # Test Lesson
    lesson = Lesson(
        id="lesson_001",
        title="MCP Introduction",
        description="Learn the basics of MCP",
        order=1,
        learning_objectives=[objective],
        exercises=[exercise],
        estimated_duration_minutes=30
    )

    # Test lesson methods
    retrieved_exercise = lesson.get_exercise_by_id("ex_001")
    assert retrieved_exercise is not None
    assert retrieved_exercise.title == "Hello MCP"

    # Test CourseModule
    module = CourseModule(
        id="module_001",
        title="MCP Fundamentals",
        description="Introduction to Model Context Protocol",
        order=1,
        lessons=[lesson],
        learning_objectives=[objective],
        difficulty_level="beginner"
    )

    # Test module methods
    module.add_lesson(lesson)
    retrieved_lesson = module.get_lesson_by_id("lesson_001")
    assert retrieved_lesson is not None

    next_lesson = module.get_next_lesson("lesson_001")
    assert next_lesson is None  # No next lesson

    all_exercises = module.get_all_exercises()
    assert len(all_exercises) == 1

    # Test serialization
    module_dict = module.to_dict()
    assert module_dict["id"] == "module_001"
    assert len(module_dict["lessons"]) == 1


def test_content_loader():
    """Test content loading system."""


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


def test_navigation():
    """Test navigation system."""


    # Create mock content loader
    class MockContentLoader:
        def get_available_modules(self):
            return ["module_001", "module_002"]

        def load_module(self, module_id):


            if module_id == "module_001":
                lesson1 = Lesson(
                    id="lesson_001", title="Lesson 1", description="First lesson",
                    order=1, exercises=[
                        Exercise(id="ex_001", title="Ex 1", description="Exercise 1",
                               difficulty="easy", estimated_minutes=10)
                    ]
                )
                lesson2 = Lesson(
                    id="lesson_002", title="Lesson 2", description="Second lesson",
                    order=2, exercises=[]
                )
                return CourseModule(
                    id="module_001", title="Module 1", description="First module",
                    order=1, lessons=[lesson1, lesson2]
                )
            elif module_id == "module_002":
                lesson3 = Lesson(
                    id="lesson_003", title="Lesson 3", description="Third lesson",
                    order=1, exercises=[]
                )
                return CourseModule(
                    id="module_002", title="Module 2", description="Second module",
                    order=2, lessons=[lesson3], prerequisites=["module_001"]
                )
            return None

    navigator = CourseNavigator(MockContentLoader())

    # Test navigation context
    context = navigator.get_navigation_context()
    assert context.current_module_id == "module_001"
    assert context.current_lesson_id == "lesson_001"

    # Test next lesson navigation
    next_result = navigator.navigate_to_next_lesson(context)
    assert next_result.success
    assert next_result.target_lesson_id == "lesson_002"

    # Test course summary
    summary = navigator.get_course_summary(context)
    assert summary["total_modules"] == 2
    assert summary["total_lessons"] == 3





