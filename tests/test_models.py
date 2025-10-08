"""Tests for course models."""

import pytest

from mcp_course.models import CourseModule, Exercise, LearningObjective, Lesson


def test_learning_objective():
    """Test LearningObjective model."""
    objective = LearningObjective(
        id="obj_001",
        description="Understand MCP basics",
        level="beginner",
        skills=["mcp", "protocol"]
    )

    assert objective.id == "obj_001"
    assert objective.description == "Understand MCP basics"
    assert objective.level == "beginner"
    assert "mcp" in objective.skills


def test_exercise():
    """Test Exercise model."""
    exercise = Exercise(
        id="ex_001",
        title="Hello MCP",
        description="Create your first MCP server",
        difficulty="easy",
        estimated_minutes=15,
        code_template="# Your code here",
        hints=["Start with imports", "Define server class"]
    )

    assert exercise.id == "ex_001"
    assert exercise.title == "Hello MCP"
    assert exercise.difficulty == "easy"
    assert exercise.estimated_minutes == 15
    assert len(exercise.hints) == 2


def test_lesson():
    """Test Lesson model."""
    objective = LearningObjective(
        id="obj_001",
        description="Understand MCP basics",
        level="beginner"
    )

    exercise = Exercise(
        id="ex_001",
        title="Hello MCP",
        description="Create your first MCP server",
        difficulty="easy",
        estimated_minutes=15
    )

    lesson = Lesson(
        id="lesson_001",
        title="MCP Introduction",
        description="Learn the basics of MCP",
        order=1,
        learning_objectives=[objective],
        exercises=[exercise],
        estimated_duration_minutes=30
    )

    assert lesson.id == "lesson_001"
    assert lesson.title == "MCP Introduction"
    assert lesson.order == 1
    assert len(lesson.learning_objectives) == 1
    assert len(lesson.exercises) == 1

    # Test exercise retrieval
    retrieved_exercise = lesson.get_exercise_by_id("ex_001")
    assert retrieved_exercise is not None
    assert retrieved_exercise.title == "Hello MCP"

    # Test adding exercise
    new_exercise = Exercise(
        id="ex_002",
        title="MCP Server",
        description="Build a server",
        difficulty="medium",
        estimated_minutes=20
    )
    lesson.add_exercise(new_exercise)
    assert len(lesson.exercises) == 2


def test_course_module():
    """Test CourseModule model."""
    objective = LearningObjective(
        id="obj_001",
        description="Understand MCP basics",
        level="beginner"
    )

    lesson = Lesson(
        id="lesson_001",
        title="MCP Introduction",
        description="Learn the basics of MCP",
        order=1,
        estimated_duration_minutes=30
    )

    module = CourseModule(
        id="module_001",
        title="MCP Fundamentals",
        description="Introduction to Model Context Protocol",
        order=1,
        lessons=[lesson],
        learning_objectives=[objective],
        difficulty_level="beginner"
    )

    assert module.id == "module_001"
    assert module.title == "MCP Fundamentals"
    assert module.order == 1
    assert module.difficulty_level == "beginner"
    assert len(module.lessons) == 1
    assert len(module.learning_objectives) == 1

    # Test lesson retrieval
    retrieved_lesson = module.get_lesson_by_id("lesson_001")
    assert retrieved_lesson is not None
    assert retrieved_lesson.title == "MCP Introduction"

    # Test lesson by order
    lesson_by_order = module.get_lesson_by_order(1)
    assert lesson_by_order is not None
    assert lesson_by_order.id == "lesson_001"

    # Test navigation
    next_lesson = module.get_next_lesson("lesson_001")
    assert next_lesson is None  # No next lesson

    prev_lesson = module.get_previous_lesson("lesson_001")
    assert prev_lesson is None  # No previous lesson

    # Test adding lesson
    lesson2 = Lesson(
        id="lesson_002",
        title="Advanced MCP",
        description="Advanced concepts",
        order=2,
        estimated_duration_minutes=45
    )
    module.add_lesson(lesson2)
    assert len(module.lessons) == 2

    # Test navigation with multiple lessons
    next_lesson = module.get_next_lesson("lesson_001")
    assert next_lesson is not None
    assert next_lesson.id == "lesson_002"

    prev_lesson = module.get_previous_lesson("lesson_002")
    assert prev_lesson is not None
    assert prev_lesson.id == "lesson_001"


def test_lesson_validation():
    """Test lesson validation."""
    with pytest.raises(ValueError, match="Lesson order must be non-negative"):
        Lesson(
            id="lesson_001",
            title="Test Lesson",
            description="Test",
            order=-1
        )

    with pytest.raises(ValueError, match="Estimated duration must be positive"):
        Lesson(
            id="lesson_001",
            title="Test Lesson",
            description="Test",
            order=1,
            estimated_duration_minutes=0
        )


def test_module_validation():
    """Test module validation."""
    with pytest.raises(ValueError, match="Module order must be non-negative"):
        CourseModule(
            id="module_001",
            title="Test Module",
            description="Test",
            order=-1
        )
