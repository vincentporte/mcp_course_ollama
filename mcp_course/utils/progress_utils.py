"""Utilities for course progress serialization and management."""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..models.progress import CourseProgress, ExerciseCompletion
from ..utils.logging import get_logger

logger = get_logger(__name__)


def serialize_progress_to_json(progress: CourseProgress) -> str:
    """Serialize CourseProgress object to JSON string."""
    try:
        return json.dumps(progress.to_dict(), indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error(f"Error serializing progress: {e}")
        raise


def deserialize_progress_from_json(json_str: str) -> CourseProgress:
    """Deserialize CourseProgress object from JSON string."""
    try:
        data = json.loads(json_str)
        return CourseProgress.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Error deserializing progress: {e}")
        raise


def export_progress_to_file(progress: CourseProgress, file_path: Path) -> None:
    """Export CourseProgress to a JSON file."""
    try:
        json_data = serialize_progress_to_json(progress)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        logger.info(f"Progress exported to {file_path}")
    except OSError as e:
        logger.error(f"Error exporting progress to file: {e}")
        raise


def import_progress_from_file(file_path: Path) -> CourseProgress:
    """Import CourseProgress from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        progress = deserialize_progress_from_json(json_str)
        logger.info(f"Progress imported from {file_path}")
        return progress
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Error importing progress from file: {e}")
        raise


def merge_progress_data(
    existing: CourseProgress, 
    new_data: CourseProgress
) -> CourseProgress:
    """Merge two CourseProgress objects, keeping the most recent data."""
    if existing.user_id != new_data.user_id or existing.module_id != new_data.module_id:
        raise ValueError("Cannot merge progress data for different users or modules")
    
    # Use the most recent last_accessed time to determine which is newer
    if new_data.last_accessed > existing.last_accessed:
        base_progress = new_data
        other_progress = existing
    else:
        base_progress = existing
        other_progress = new_data
    
    # Merge exercise completions, keeping the most recent for each exercise
    merged_exercises = {}
    
    # Add all exercises from both progress objects
    for exercise in base_progress.practical_exercises + other_progress.practical_exercises:
        exercise_id = exercise.exercise_id
        if (exercise_id not in merged_exercises or 
            (exercise.completion_time and 
             (not merged_exercises[exercise_id].completion_time or
              exercise.completion_time > merged_exercises[exercise_id].completion_time))):
            merged_exercises[exercise_id] = exercise
    
    # Create merged progress object
    merged_progress = CourseProgress(
        user_id=base_progress.user_id,
        module_id=base_progress.module_id,
        completion_status=base_progress.completion_status,
        assessment_score=base_progress.assessment_score,
        last_accessed=max(existing.last_accessed, new_data.last_accessed),
        practical_exercises=list(merged_exercises.values()),
        notes=base_progress.notes or other_progress.notes,
        time_spent_minutes=max(existing.time_spent_minutes, new_data.time_spent_minutes)
    )
    
    logger.info(f"Merged progress data for user {merged_progress.user_id}, module {merged_progress.module_id}")
    return merged_progress


def calculate_overall_progress(user_progress: Dict[str, CourseProgress]) -> Dict[str, Any]:
    """Calculate overall progress statistics for a user across all modules."""
    if not user_progress:
        return {
            "total_modules": 0,
            "completed_modules": 0,
            "in_progress_modules": 0,
            "not_started_modules": 0,
            "overall_completion_percentage": 0.0,
            "total_exercises": 0,
            "completed_exercises": 0,
            "total_time_spent_minutes": 0,
            "average_assessment_score": None
        }
    
    total_modules = len(user_progress)
    completed_modules = sum(1 for p in user_progress.values() if p.completion_status == 'completed')
    in_progress_modules = sum(1 for p in user_progress.values() if p.completion_status == 'in_progress')
    not_started_modules = sum(1 for p in user_progress.values() if p.completion_status == 'not_started')
    
    total_exercises = sum(len(p.practical_exercises) for p in user_progress.values())
    completed_exercises = sum(
        sum(1 for ex in p.practical_exercises if ex.completed) 
        for p in user_progress.values()
    )
    
    total_time_spent = sum(p.time_spent_minutes for p in user_progress.values())
    
    # Calculate average assessment score (only for modules with scores)
    scores = [p.assessment_score for p in user_progress.values() if p.assessment_score is not None]
    average_score = sum(scores) / len(scores) if scores else None
    
    overall_completion = (completed_modules / total_modules) * 100.0 if total_modules > 0 else 0.0
    
    return {
        "total_modules": total_modules,
        "completed_modules": completed_modules,
        "in_progress_modules": in_progress_modules,
        "not_started_modules": not_started_modules,
        "overall_completion_percentage": overall_completion,
        "total_exercises": total_exercises,
        "completed_exercises": completed_exercises,
        "exercise_completion_percentage": (completed_exercises / total_exercises) * 100.0 if total_exercises > 0 else 0.0,
        "total_time_spent_minutes": total_time_spent,
        "average_assessment_score": average_score
    }


def create_progress_summary(progress: CourseProgress) -> Dict[str, Any]:
    """Create a summary of progress for a specific module."""
    completed_exercises = sum(1 for ex in progress.practical_exercises if ex.completed)
    total_exercises = len(progress.practical_exercises)
    
    return {
        "user_id": progress.user_id,
        "module_id": progress.module_id,
        "completion_status": progress.completion_status,
        "assessment_score": progress.assessment_score,
        "last_accessed": progress.last_accessed.isoformat(),
        "exercise_completion": {
            "completed": completed_exercises,
            "total": total_exercises,
            "percentage": progress.calculate_completion_percentage()
        },
        "time_spent_minutes": progress.time_spent_minutes,
        "has_notes": bool(progress.notes)
    }