"""Navigation utilities for sequential learning progression."""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from mcp_course.models.content import CourseModule, Lesson, ContentLoader
from mcp_course.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NavigationContext:
    """Context information for course navigation."""
    
    current_module_id: Optional[str] = None
    current_lesson_id: Optional[str] = None
    available_modules: List[str] = None
    
    def __post_init__(self):
        if self.available_modules is None:
            self.available_modules = []


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
    
    def get_navigation_context(self) -> NavigationContext:
        """Get current navigation context."""
        available_modules = self.content_loader.get_available_modules()
        
        # Start with first module and first lesson if available
        current_module_id = None
        current_lesson_id = None
        
        if available_modules:
            current_module_id = available_modules[0]
            module = self.content_loader.load_module(current_module_id)
            if module and module.lessons:
                current_lesson_id = module.lessons[0].id
        
        return NavigationContext(
            current_module_id=current_module_id,
            current_lesson_id=current_lesson_id,
            available_modules=available_modules
        )
    
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
    
    def get_learning_path(self, context: NavigationContext) -> List[Tuple[str, str]]:
        """Get the complete learning path as (module_id, lesson_id) tuples."""
        path = []
        
        for module_id in context.available_modules:
            module = self.content_loader.load_module(module_id)
            if module:
                for lesson in sorted(module.lessons, key=lambda l: l.order):
                    path.append((module_id, lesson.id))
        
        return path
    
    def get_course_summary(self, context: NavigationContext) -> Dict[str, Any]:
        """Get a summary of the course structure."""
        total_modules = len(context.available_modules)
        total_lessons = 0
        
        for module_id in context.available_modules:
            module = self.content_loader.load_module(module_id)
            if module:
                total_lessons += len(module.lessons)
        
        return {
            "total_modules": total_modules,
            "total_lessons": total_lessons,
            "current_module": context.current_module_id,
            "current_lesson": context.current_lesson_id,
            "available_modules": context.available_modules
        }