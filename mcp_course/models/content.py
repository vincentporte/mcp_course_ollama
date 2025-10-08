"""Course content structure models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mcp_course.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class LearningObjective:
    """Represents a specific learning objective for a lesson or module."""

    id: str
    description: str
    level: Literal['beginner', 'intermediate', 'advanced']
    skills: list[str] = field(default_factory=list)


@dataclass
class Exercise:
    """Represents a practical exercise within a lesson."""

    id: str
    title: str
    description: str
    difficulty: Literal['easy', 'medium', 'hard']
    estimated_minutes: int
    code_template: str | None = None
    solution: str | None = None
    hints: list[str] = field(default_factory=list)
    validation_criteria: list[str] = field(default_factory=list)


@dataclass
class Lesson:
    """Represents an individual lesson within a course module."""

    id: str
    title: str
    description: str
    order: int
    content_path: Path | None = None
    learning_objectives: list[LearningObjective] = field(default_factory=list)
    exercises: list[Exercise] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    estimated_duration_minutes: int = 30
    content_type: Literal['text', 'interactive', 'video', 'mixed'] = 'text'

    def __post_init__(self):
        """Validate lesson data after initialization."""
        if self.order < 0:
            raise ValueError("Lesson order must be non-negative")
        if self.estimated_duration_minutes <= 0:
            raise ValueError("Estimated duration must be positive")

    def get_exercise_by_id(self, exercise_id: str) -> Exercise | None:
        """Get exercise by ID."""
        for exercise in self.exercises:
            if exercise.id == exercise_id:
                return exercise
        return None

    def add_exercise(self, exercise: Exercise) -> None:
        """Add an exercise to the lesson."""
        # Remove existing exercise with same ID
        self.exercises = [ex for ex in self.exercises if ex.id != exercise.id]
        self.exercises.append(exercise)

    def to_dict(self) -> dict[str, Any]:
        """Convert lesson to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "order": self.order,
            "content_path": str(self.content_path) if self.content_path else None,
            "learning_objectives": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "level": obj.level,
                    "skills": obj.skills
                }
                for obj in self.learning_objectives
            ],
            "exercises": [
                {
                    "id": ex.id,
                    "title": ex.title,
                    "description": ex.description,
                    "difficulty": ex.difficulty,
                    "estimated_minutes": ex.estimated_minutes,
                    "code_template": ex.code_template,
                    "solution": ex.solution,
                    "hints": ex.hints,
                    "validation_criteria": ex.validation_criteria
                }
                for ex in self.exercises
            ],
            "prerequisites": self.prerequisites,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "content_type": self.content_type
        }


@dataclass
class CourseModule:
    """Represents a course module containing multiple lessons."""

    id: str
    title: str
    description: str
    order: int
    lessons: list[Lesson] = field(default_factory=list)
    learning_objectives: list[LearningObjective] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    estimated_duration_minutes: int = 0
    difficulty_level: Literal['beginner', 'intermediate', 'advanced'] = 'beginner'

    def __post_init__(self):
        """Calculate total duration and validate module data."""
        if self.order < 0:
            raise ValueError("Module order must be non-negative")

        # Calculate total estimated duration from lessons
        if not self.estimated_duration_minutes and self.lessons:
            self.estimated_duration_minutes = sum(
                lesson.estimated_duration_minutes for lesson in self.lessons
            )

    def add_lesson(self, lesson: Lesson) -> None:
        """Add a lesson to the module."""
        # Remove existing lesson with same ID
        self.lessons = [existing_lesson for existing_lesson in self.lessons if existing_lesson.id != lesson.id]
        self.lessons.append(lesson)

        # Sort lessons by order
        self.lessons.sort(key=lambda lesson: lesson.order)

        # Recalculate total duration
        self.estimated_duration_minutes = sum(
            lesson.estimated_duration_minutes for lesson in self.lessons
        )

    def get_lesson_by_id(self, lesson_id: str) -> Lesson | None:
        """Get lesson by ID."""
        for lesson in self.lessons:
            if lesson.id == lesson_id:
                return lesson
        return None

    def get_lesson_by_order(self, order: int) -> Lesson | None:
        """Get lesson by order number."""
        for lesson in self.lessons:
            if lesson.order == order:
                return lesson
        return None

    def get_next_lesson(self, current_lesson_id: str) -> Lesson | None:
        """Get the next lesson after the current one."""
        current_lesson = self.get_lesson_by_id(current_lesson_id)
        if not current_lesson:
            return None

        # Find lesson with next order number
        next_order = current_lesson.order + 1
        return self.get_lesson_by_order(next_order)

    def get_previous_lesson(self, current_lesson_id: str) -> Lesson | None:
        """Get the previous lesson before the current one."""
        current_lesson = self.get_lesson_by_id(current_lesson_id)
        if not current_lesson:
            return None

        # Find lesson with previous order number
        prev_order = current_lesson.order - 1
        return self.get_lesson_by_order(prev_order)

    def get_all_exercises(self) -> list[Exercise]:
        """Get all exercises from all lessons in the module."""
        exercises = []
        for lesson in self.lessons:
            exercises.extend(lesson.exercises)
        return exercises

    def to_dict(self) -> dict[str, Any]:
        """Convert module to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "order": self.order,
            "lessons": [lesson.to_dict() for lesson in self.lessons],
            "learning_objectives": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "level": obj.level,
                    "skills": obj.skills
                }
                for obj in self.learning_objectives
            ],
            "prerequisites": self.prerequisites,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "difficulty_level": self.difficulty_level
        }


class ContentLoader(ABC):
    """Abstract base class for loading course content."""

    @abstractmethod
    def load_module(self, module_id: str) -> CourseModule | None:
        """Load a course module by ID."""
        pass

    @abstractmethod
    def load_lesson(self, module_id: str, lesson_id: str) -> Lesson | None:
        """Load a specific lesson."""
        pass

    @abstractmethod
    def get_available_modules(self) -> list[str]:
        """Get list of available module IDs."""
        pass
