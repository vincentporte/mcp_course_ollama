"""Navigation utilities for sequential learning progression."""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..models.content import CourseModule, Lesson, ContentLoader
from ..models.progress import CourseProgress
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NavigationContext:
    """Context information for course navigation."""
    
    current_module_id: Optional[str] = None
    current_lesson_id: Optional[str] = None
    available_modules: List[str] = None
    module_progress: Dict[str, CourseProgress] = None
    
    def __post_init__(self):
        if self.available_modules is None:
            self.available_modules = []
        if self.module_progress is None:
            self.module_progress = {}


@dataclass
class NavigationResult:
    """Result of a navigation operation."""
    
    success: bool
    target_module_id: Optional[str] = None
    target_lesson_id: Optional[str] = None
    message: Optional[str] = None
    prerequisites_missing: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites_missing is None:
            self.prerequisites_missing = []


class CourseNavigator:
    """Handles navigation through course modules and lessons."""
    
    def __init__(self, content_loader: ContentLoader):
        """Initialize navigator with content loader."""
        self.content_loader = content_loader
        logger.info("Initialized course navigator")
    
    def get_navigation_context(self, user_progress: Dict[str, CourseProgress]) -> NavigationContext:
        """Get current navigation context for a user."""
        available_modules = self.content_loader.get_available_modules()
        
        # Find current position based on progress
        current_module_id = None
        current_lesson_id = None
        
        # Look for the first incomplete module
        for module_id in available_modules:
            progress = user_progress.get(module_id)
            if not progress or progress.completion_status != 'completed':
                current_module_id = module_id
                
                # Find current lesson within module
                module = self.content_loader.load_module(module_id)
                if module and progress:
                    current_lesson_id = self._find_current_lesson(module, progress)
                elif module:
                    # Start with first lesson if no progress
                    if module.lessons:
                        current_lesson_id = module.lessons[0].id
                break
        
        return NavigationContext(
            current_module_id=current_module_id,
            current_lesson_id=current_lesson_id,
            available_modules=available_modules,
            module_progress=user_progress
        )
    
    def _find_current_lesson(self, module: CourseModule, progress: CourseProgress) -> Optional[str]:
        """Find the current lesson based on progress."""
        if not module.lessons:
            return None
        
        # Find first incomplete lesson
        for lesson in sorted(module.lessons, key=lambda l: l.order):
            lesson_exercises = [ex for ex in progress.practical_exercises 
                             if any(lesson_ex.id == ex.exercise_id for lesson_ex in lesson.exercises)]
            
            # If lesson has exercises and not all are completed
            if lesson.exercises:
                completed_exercises = [ex for ex in lesson_exercises if ex.completed]
                if len(completed_exercises) < len(lesson.exercises):
                    return lesson.id
            else:
                # For lessons without exercises, check if module is in progress
                if progress.completion_status == 'in_progress':
                    return lesson.id
        
        # If all lessons seem complete but module isn't, return last lesson
        if progress.completion_status != 'completed' and module.lessons:
            return module.lessons[-1].id
        
        return None
    
    def navigate_to_next_lesson(self, context: NavigationContext) -> NavigationResult:
        """Navigate to the next lesson in sequence."""
        if not context.current_module_id:
            return NavigationResult(
                success=False,
                message="No current module to navigate from"
            )
        
        module = self.content_loader.load_module(context.current_module_id)
        if not module:
            return NavigationResult(
                success=False,
                message=f"Could not load module: {context.current_module_id}"
            )
        
        if context.current_lesson_id:
            # Try to get next lesson in current module
            next_lesson = module.get_next_lesson(context.current_lesson_id)
            if next_lesson:
                # Check prerequisites
                missing_prereqs = self._check_lesson_prerequisites(
                    next_lesson, context.module_progress
                )
                if missing_prereqs:
                    return NavigationResult(
                        success=False,
                        message="Prerequisites not met for next lesson",
                        prerequisites_missing=missing_prereqs
                    )
                
                return NavigationResult(
                    success=True,
                    target_module_id=context.current_module_id,
                    target_lesson_id=next_lesson.id,
                    message=f"Navigated to lesson: {next_lesson.title}"
                )
        
        # No next lesson in current module, try next module
        return self._navigate_to_next_module(context)
    
    def navigate_to_previous_lesson(self, context: NavigationContext) -> NavigationResult:
        """Navigate to the previous lesson in sequence."""
        if not context.current_module_id or not context.current_lesson_id:
            return NavigationResult(
                success=False,
                message="No current lesson to navigate from"
            )
        
        module = self.content_loader.load_module(context.current_module_id)
        if not module:
            return NavigationResult(
                success=False,
                message=f"Could not load module: {context.current_module_id}"
            )
        
        # Try to get previous lesson in current module
        prev_lesson = module.get_previous_lesson(context.current_lesson_id)
        if prev_lesson:
            return NavigationResult(
                success=True,
                target_module_id=context.current_module_id,
                target_lesson_id=prev_lesson.id,
                message=f"Navigated to lesson: {prev_lesson.title}"
            )
        
        # No previous lesson in current module, try previous module
        return self._navigate_to_previous_module(context)
    
    def _navigate_to_next_module(self, context: NavigationContext) -> NavigationResult:
        """Navigate to the first lesson of the next module."""
        if not context.current_module_id:
            return NavigationResult(success=False, message="No current module")
        
        try:
            current_index = context.available_modules.index(context.current_module_id)
            if current_index + 1 < len(context.available_modules):
                next_module_id = context.available_modules[current_index + 1]
                next_module = self.content_loader.load_module(next_module_id)
                
                if not next_module:
                    return NavigationResult(
                        success=False,
                        message=f"Could not load next module: {next_module_id}"
                    )
                
                # Check module prerequisites
                missing_prereqs = self._check_module_prerequisites(
                    next_module, context.module_progress
                )
                if missing_prereqs:
                    return NavigationResult(
                        success=False,
                        message="Prerequisites not met for next module",
                        prerequisites_missing=missing_prereqs
                    )
                
                # Get first lesson
                if next_module.lessons:
                    first_lesson = min(next_module.lessons, key=lambda l: l.order)
                    return NavigationResult(
                        success=True,
                        target_module_id=next_module_id,
                        target_lesson_id=first_lesson.id,
                        message=f"Navigated to module: {next_module.title}"
                    )
                else:
                    return NavigationResult(
                        success=True,
                        target_module_id=next_module_id,
                        target_lesson_id=None,
                        message=f"Navigated to module: {next_module.title} (no lessons)"
                    )
            else:
                return NavigationResult(
                    success=False,
                    message="Already at the last module"
                )
        except ValueError:
            return NavigationResult(
                success=False,
                message="Current module not found in available modules"
            )
    
    def _navigate_to_previous_module(self, context: NavigationContext) -> NavigationResult:
        """Navigate to the last lesson of the previous module."""
        if not context.current_module_id:
            return NavigationResult(success=False, message="No current module")
        
        try:
            current_index = context.available_modules.index(context.current_module_id)
            if current_index > 0:
                prev_module_id = context.available_modules[current_index - 1]
                prev_module = self.content_loader.load_module(prev_module_id)
                
                if not prev_module:
                    return NavigationResult(
                        success=False,
                        message=f"Could not load previous module: {prev_module_id}"
                    )
                
                # Get last lesson
                if prev_module.lessons:
                    last_lesson = max(prev_module.lessons, key=lambda l: l.order)
                    return NavigationResult(
                        success=True,
                        target_module_id=prev_module_id,
                        target_lesson_id=last_lesson.id,
                        message=f"Navigated to module: {prev_module.title}"
                    )
                else:
                    return NavigationResult(
                        success=True,
                        target_module_id=prev_module_id,
                        target_lesson_id=None,
                        message=f"Navigated to module: {prev_module.title} (no lessons)"
                    )
            else:
                return NavigationResult(
                    success=False,
                    message="Already at the first module"
                )
        except ValueError:
            return NavigationResult(
                success=False,
                message="Current module not found in available modules"
            )
    
    def _check_module_prerequisites(self, module: CourseModule, user_progress: Dict[str, CourseProgress]) -> List[str]:
        """Check if module prerequisites are met."""
        missing_prereqs = []
        
        for prereq_module_id in module.prerequisites:
            progress = user_progress.get(prereq_module_id)
            if not progress or progress.completion_status != 'completed':
                missing_prereqs.append(prereq_module_id)
        
        return missing_prereqs
    
    def _check_lesson_prerequisites(self, lesson: Lesson, user_progress: Dict[str, CourseProgress]) -> List[str]:
        """Check if lesson prerequisites are met."""
        missing_prereqs = []
        
        for prereq_lesson_id in lesson.prerequisites:
            # Prerequisites can be lesson IDs or module IDs
            found = False
            
            # Check if it's a module prerequisite
            if prereq_lesson_id in user_progress:
                progress = user_progress[prereq_lesson_id]
                if progress.completion_status == 'completed':
                    found = True
            else:
                # Check if it's a lesson within any module
                for module_id, progress in user_progress.items():
                    for exercise in progress.practical_exercises:
                        if exercise.exercise_id.startswith(prereq_lesson_id) and exercise.completed:
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                missing_prereqs.append(prereq_lesson_id)
        
        return missing_prereqs
    
    def get_learning_path(self, context: NavigationContext) -> List[Tuple[str, str]]:
        """Get the complete learning path as (module_id, lesson_id) tuples."""
        path = []
        
        for module_id in context.available_modules:
            module = self.content_loader.load_module(module_id)
            if module:
                for lesson in sorted(module.lessons, key=lambda l: l.order):
                    path.append((module_id, lesson.id))
        
        return path
    
    def get_progress_summary(self, context: NavigationContext) -> Dict[str, Any]:
        """Get a summary of learning progress."""
        total_modules = len(context.available_modules)
        completed_modules = sum(
            1 for module_id in context.available_modules
            if context.module_progress.get(module_id, {}).completion_status == 'completed'
        )
        
        total_lessons = 0
        completed_lessons = 0
        
        for module_id in context.available_modules:
            module = self.content_loader.load_module(module_id)
            if module:
                total_lessons += len(module.lessons)
                
                progress = context.module_progress.get(module_id)
                if progress:
                    # Count completed lessons based on exercise completion
                    for lesson in module.lessons:
                        lesson_exercises = [ex.id for ex in lesson.exercises]
                        if lesson_exercises:
                            completed_exercises = [
                                ex for ex in progress.practical_exercises
                                if ex.exercise_id in lesson_exercises and ex.completed
                            ]
                            if len(completed_exercises) == len(lesson_exercises):
                                completed_lessons += 1
                        elif progress.completion_status == 'completed':
                            # Lesson without exercises in completed module
                            completed_lessons += 1
        
        return {
            "total_modules": total_modules,
            "completed_modules": completed_modules,
            "module_completion_percentage": (completed_modules / total_modules) * 100 if total_modules > 0 else 0,
            "total_lessons": total_lessons,
            "completed_lessons": completed_lessons,
            "lesson_completion_percentage": (completed_lessons / total_lessons) * 100 if total_lessons > 0 else 0,
            "current_module": context.current_module_id,
            "current_lesson": context.current_lesson_id
        }