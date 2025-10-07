"""Course progress tracking models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
import json
from pathlib import Path


@dataclass
class ExerciseCompletion:
    """Tracks completion status and details for individual exercises."""
    
    exercise_id: str
    completed: bool
    code_submission: Optional[str] = None
    feedback: Optional[str] = None
    completion_time: Optional[datetime] = None
    attempts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exercise_id": self.exercise_id,
            "completed": self.completed,
            "code_submission": self.code_submission,
            "feedback": self.feedback,
            "completion_time": self.completion_time.isoformat() if self.completion_time else None,
            "attempts": self.attempts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExerciseCompletion":
        """Create instance from dictionary."""
        completion_time = None
        if data.get("completion_time"):
            completion_time = datetime.fromisoformat(data["completion_time"])
            
        return cls(
            exercise_id=data["exercise_id"],
            completed=data["completed"],
            code_submission=data.get("code_submission"),
            feedback=data.get("feedback"),
            completion_time=completion_time,
            attempts=data.get("attempts", 0)
        )


@dataclass
class CourseProgress:
    """Tracks overall progress through course modules and exercises."""
    
    user_id: str
    module_id: str
    completion_status: Literal['not_started', 'in_progress', 'completed']
    assessment_score: Optional[int] = None
    last_accessed: datetime = field(default_factory=datetime.now)
    practical_exercises: List[ExerciseCompletion] = field(default_factory=list)
    notes: Optional[str] = None
    time_spent_minutes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "module_id": self.module_id,
            "completion_status": self.completion_status,
            "assessment_score": self.assessment_score,
            "last_accessed": self.last_accessed.isoformat(),
            "practical_exercises": [ex.to_dict() for ex in self.practical_exercises],
            "notes": self.notes,
            "time_spent_minutes": self.time_spent_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CourseProgress":
        """Create instance from dictionary."""
        exercises = [
            ExerciseCompletion.from_dict(ex_data) 
            for ex_data in data.get("practical_exercises", [])
        ]
        
        return cls(
            user_id=data["user_id"],
            module_id=data["module_id"],
            completion_status=data["completion_status"],
            assessment_score=data.get("assessment_score"),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            practical_exercises=exercises,
            notes=data.get("notes"),
            time_spent_minutes=data.get("time_spent_minutes", 0)
        )
    
    def add_exercise_completion(self, exercise: ExerciseCompletion) -> None:
        """Add or update exercise completion."""
        # Remove existing completion for same exercise
        self.practical_exercises = [
            ex for ex in self.practical_exercises 
            if ex.exercise_id != exercise.exercise_id
        ]
        self.practical_exercises.append(exercise)
        self.last_accessed = datetime.now()
    
    def get_exercise_completion(self, exercise_id: str) -> Optional[ExerciseCompletion]:
        """Get completion status for specific exercise."""
        for exercise in self.practical_exercises:
            if exercise.exercise_id == exercise_id:
                return exercise
        return None
    
    def calculate_completion_percentage(self) -> float:
        """Calculate percentage of exercises completed."""
        if not self.practical_exercises:
            return 0.0
        
        completed_count = sum(1 for ex in self.practical_exercises if ex.completed)
        return (completed_count / len(self.practical_exercises)) * 100.0