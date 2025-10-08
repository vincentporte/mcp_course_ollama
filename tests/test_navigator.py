"""Tests for course navigation system."""

from mcp_course.content import CourseNavigator
from mcp_course.models import CourseModule, Exercise, Lesson


class MockContentLoader:
    """Mock content loader for testing."""

    def get_available_modules(self):
        return ["module_001", "module_002"]

    def load_module(self, module_id):
        if module_id == "module_001":
            lesson1 = Lesson(
                id="lesson_001",
                title="Lesson 1",
                description="First lesson",
                order=1,
                exercises=[
                    Exercise(
                        id="ex_001",
                        title="Ex 1",
                        description="Exercise 1",
                        difficulty="easy",
                        estimated_minutes=10
                    )
                ]
            )
            lesson2 = Lesson(
                id="lesson_002",
                title="Lesson 2",
                description="Second lesson",
                order=2,
                exercises=[]
            )
            return CourseModule(
                id="module_001",
                title="Module 1",
                description="First module",
                order=1,
                lessons=[lesson1, lesson2]
            )
        elif module_id == "module_002":
            lesson3 = Lesson(
                id="lesson_003",
                title="Lesson 3",
                description="Third lesson",
                order=1,
                exercises=[]
            )
            return CourseModule(
                id="module_002",
                title="Module 2",
                description="Second module",
                order=2,
                lessons=[lesson3],
                prerequisites=["module_001"]
            )
        return None


def test_navigation_context():
    """Test navigation context creation."""
    navigator = CourseNavigator(MockContentLoader())

    context = navigator.get_navigation_context()
    assert context.current_module_id == "module_001"
    assert context.current_lesson_id == "lesson_001"
    assert len(context.available_modules) == 2
    assert "module_001" in context.available_modules
    assert "module_002" in context.available_modules


def test_navigate_to_next_lesson():
    """Test navigation to next lesson."""
    navigator = CourseNavigator(MockContentLoader())
    context = navigator.get_navigation_context()

    # Navigate from lesson 1 to lesson 2 in same module
    next_result = navigator.navigate_to_next_lesson(context)
    assert next_result.success
    assert next_result.target_module_id == "module_001"
    assert next_result.target_lesson_id == "lesson_002"

    # Update context and navigate to next module
    context.current_lesson_id = "lesson_002"
    next_result = navigator.navigate_to_next_lesson(context)
    assert next_result.success
    assert next_result.target_module_id == "module_002"
    assert next_result.target_lesson_id == "lesson_003"


def test_navigate_to_previous_lesson():
    """Test navigation to previous lesson."""
    navigator = CourseNavigator(MockContentLoader())
    context = navigator.get_navigation_context()

    # Start from lesson 2
    context.current_lesson_id = "lesson_002"

    # Navigate back to lesson 1
    prev_result = navigator.navigate_to_previous_lesson(context)
    assert prev_result.success
    assert prev_result.target_module_id == "module_001"
    assert prev_result.target_lesson_id == "lesson_001"


def test_navigate_at_boundaries():
    """Test navigation at course boundaries."""
    navigator = CourseNavigator(MockContentLoader())
    context = navigator.get_navigation_context()

    # Try to navigate previous from first lesson
    prev_result = navigator.navigate_to_previous_lesson(context)
    assert not prev_result.success
    assert "first module" in prev_result.message.lower()

    # Navigate to last lesson of last module
    context.current_module_id = "module_002"
    context.current_lesson_id = "lesson_003"

    # Try to navigate next from last lesson
    next_result = navigator.navigate_to_next_lesson(context)
    assert not next_result.success
    assert "last module" in next_result.message.lower()


def test_navigation_without_current_context():
    """Test navigation without current context."""
    navigator = CourseNavigator(MockContentLoader())
    context = navigator.get_navigation_context()

    # Clear current context
    context.current_module_id = None
    context.current_lesson_id = None

    next_result = navigator.navigate_to_next_lesson(context)
    assert not next_result.success
    assert "No current module" in next_result.message

    prev_result = navigator.navigate_to_previous_lesson(context)
    assert not prev_result.success
    assert "No current lesson" in prev_result.message


def test_get_learning_path():
    """Test getting complete learning path."""
    navigator = CourseNavigator(MockContentLoader())
    context = navigator.get_navigation_context()

    path = navigator.get_learning_path(context)
    expected_path = [
        ("module_001", "lesson_001"),
        ("module_001", "lesson_002"),
        ("module_002", "lesson_003")
    ]

    assert path == expected_path


def test_get_course_summary():
    """Test getting course summary."""
    navigator = CourseNavigator(MockContentLoader())
    context = navigator.get_navigation_context()

    summary = navigator.get_course_summary(context)

    assert summary["total_modules"] == 2
    assert summary["total_lessons"] == 3
    assert summary["current_module"] == "module_001"
    assert summary["current_lesson"] == "lesson_001"
    assert len(summary["available_modules"]) == 2


def test_navigation_with_empty_loader():
    """Test navigation with empty content loader."""

    class EmptyContentLoader:
        def get_available_modules(self):
            return []

        def load_module(self):
            return None

    navigator = CourseNavigator(EmptyContentLoader())
    context = navigator.get_navigation_context()

    assert context.current_module_id is None
    assert context.current_lesson_id is None
    assert len(context.available_modules) == 0

    summary = navigator.get_course_summary(context)
    assert summary["total_modules"] == 0
    assert summary["total_lessons"] == 0


def test_navigation_with_module_without_lessons():
    """Test navigation with module that has no lessons."""

    class NoLessonsContentLoader:
        def get_available_modules(self):
            return ["empty_module"]

        def load_module(self, module_id):
            if module_id == "empty_module":
                return CourseModule(
                    id="empty_module",
                    title="Empty Module",
                    description="Module with no lessons",
                    order=1,
                    lessons=[]
                )
            return None

    navigator = CourseNavigator(NoLessonsContentLoader())
    context = navigator.get_navigation_context()

    assert context.current_module_id == "empty_module"
    assert context.current_lesson_id is None

    # Try navigation - should fail gracefully
    next_result = navigator.navigate_to_next_lesson(context)
    assert not next_result.success
