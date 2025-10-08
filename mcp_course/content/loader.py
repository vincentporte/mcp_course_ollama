"""Content loading system for course modules and lessons."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

from mcp_course.models.content import CourseModule, Lesson, Exercise, LearningObjective, ContentLoader
from mcp_course.utils.logging import get_logger

logger = get_logger(__name__)


class FileSystemContentLoader(ContentLoader):
    """Loads course content from filesystem structure."""
    
    def __init__(self, content_root: Path):
        """Initialize with root directory containing course content."""
        self.content_root = Path(content_root)
        if not self.content_root.exists():
            raise ValueError(f"Content root directory does not exist: {content_root}")
        
        self._module_cache: Dict[str, CourseModule] = {}
        logger.info(f"Initialized content loader with root: {self.content_root}")
    
    def get_available_modules(self) -> List[str]:
        """Get list of available module IDs from filesystem."""
        module_dirs = []
        
        # Look for directories that contain module.json or module.yaml files
        for item in self.content_root.iterdir():
            if item.is_dir():
                module_file = self._find_module_file(item)
                if module_file:
                    module_dirs.append(item.name)
        
        # Sort by module order if possible
        try:
            module_dirs.sort(key=lambda x: self._get_module_order(x))
        except Exception:
            # Fallback to alphabetical sort
            module_dirs.sort()
        
        return module_dirs
    
    def _find_module_file(self, module_dir: Path) -> Optional[Path]:
        """Find module configuration file in directory."""
        module_file = module_dir / 'module.json'
        if module_file.exists():
            return module_file
        return None
    
    def _get_module_order(self, module_id: str) -> int:
        """Get module order from configuration or infer from name."""
        try:
            module = self.load_module(module_id)
            return module.order if module else 999
        except Exception:
            # Try to extract number from module name
            match = re.search(r'(\d+)', module_id)
            return int(match.group(1)) if match else 999
    
    def load_module(self, module_id: str) -> Optional[CourseModule]:
        """Load a course module by ID."""
        if module_id in self._module_cache:
            return self._module_cache[module_id]
        
        module_dir = self.content_root / module_id
        if not module_dir.exists() or not module_dir.is_dir():
            logger.warning(f"Module directory not found: {module_dir}")
            return None
        
        module_file = self._find_module_file(module_dir)
        if not module_file:
            logger.warning(f"Module configuration file not found in: {module_dir}")
            return None
        
        try:
            module_data = self._load_config_file(module_file)
            module = self._create_module_from_data(module_id, module_data, module_dir)
            
            # Load lessons
            lessons = self._load_module_lessons(module_dir)
            for lesson in lessons:
                module.add_lesson(lesson)
            
            self._module_cache[module_id] = module
            logger.info(f"Loaded module: {module_id} with {len(lessons)} lessons")
            return module
            
        except Exception as e:
            logger.error(f"Error loading module {module_id}: {e}")
            return None
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_module_from_data(self, module_id: str, data: Dict[str, Any], module_dir: Path) -> CourseModule:
        """Create CourseModule from configuration data."""
        learning_objectives = []
        for obj_data in data.get('learning_objectives', []):
            learning_objectives.append(LearningObjective(
                id=obj_data['id'],
                description=obj_data['description'],
                level=obj_data.get('level', 'beginner'),
                skills=obj_data.get('skills', [])
            ))
        
        return CourseModule(
            id=module_id,
            title=data['title'],
            description=data['description'],
            order=data.get('order', 0),
            learning_objectives=learning_objectives,
            prerequisites=data.get('prerequisites', []),
            estimated_duration_minutes=data.get('estimated_duration_minutes', 0),
            difficulty_level=data.get('difficulty_level', 'beginner')
        )
    
    def _load_module_lessons(self, module_dir: Path) -> List[Lesson]:
        """Load all lessons for a module."""
        lessons = []
        lessons_dir = module_dir / 'lessons'
        
        if not lessons_dir.exists():
            logger.warning(f"No lessons directory found in: {module_dir}")
            return lessons
        
        # Find all lesson directories or files
        for item in lessons_dir.iterdir():
            if item.is_dir():
                lesson = self._load_lesson_from_dir(item)
                if lesson:
                    lessons.append(lesson)
            elif item.suffix.lower() == '.json':
                lesson = self._load_lesson_from_file(item)
                if lesson:
                    lessons.append(lesson)
        
        # Sort lessons by order
        lessons.sort(key=lambda l: l.order)
        return lessons
    
    def _load_lesson_from_dir(self, lesson_dir: Path) -> Optional[Lesson]:
        """Load lesson from directory structure."""
        lesson_file = lesson_dir / 'lesson.json'
        if not lesson_file.exists():
            lesson_file = None
        
        if not lesson_file:
            logger.warning(f"No lesson configuration found in: {lesson_dir}")
            return None
        
        try:
            lesson_data = self._load_config_file(lesson_file)
            lesson_id = lesson_data.get('id', lesson_dir.name)
            
            # Look for content file
            content_path = None
            for content_file in ['content.md', 'content.txt', 'content.html']:
                candidate = lesson_dir / content_file
                if candidate.exists():
                    content_path = candidate
                    break
            
            return self._create_lesson_from_data(lesson_id, lesson_data, content_path)
            
        except Exception as e:
            logger.error(f"Error loading lesson from {lesson_dir}: {e}")
            return None
    
    def _load_lesson_from_file(self, lesson_file: Path) -> Optional[Lesson]:
        """Load lesson from single configuration file."""
        try:
            lesson_data = self._load_config_file(lesson_file)
            lesson_id = lesson_data.get('id', lesson_file.stem)
            
            # Content path might be specified in config
            content_path = None
            if 'content_path' in lesson_data:
                content_path = lesson_file.parent / lesson_data['content_path']
            
            return self._create_lesson_from_data(lesson_id, lesson_data, content_path)
            
        except Exception as e:
            logger.error(f"Error loading lesson from {lesson_file}: {e}")
            return None
    
    def _create_lesson_from_data(self, lesson_id: str, data: Dict[str, Any], content_path: Optional[Path]) -> Lesson:
        """Create Lesson from configuration data."""
        # Load learning objectives
        learning_objectives = []
        for obj_data in data.get('learning_objectives', []):
            learning_objectives.append(LearningObjective(
                id=obj_data['id'],
                description=obj_data['description'],
                level=obj_data.get('level', 'beginner'),
                skills=obj_data.get('skills', [])
            ))
        
        # Load exercises
        exercises = []
        for ex_data in data.get('exercises', []):
            exercises.append(Exercise(
                id=ex_data['id'],
                title=ex_data['title'],
                description=ex_data['description'],
                difficulty=ex_data.get('difficulty', 'medium'),
                estimated_minutes=ex_data.get('estimated_minutes', 15),
                code_template=ex_data.get('code_template'),
                solution=ex_data.get('solution'),
                hints=ex_data.get('hints', []),
                validation_criteria=ex_data.get('validation_criteria', [])
            ))
        
        return Lesson(
            id=lesson_id,
            title=data['title'],
            description=data['description'],
            order=data.get('order', 0),
            content_path=content_path,
            learning_objectives=learning_objectives,
            exercises=exercises,
            prerequisites=data.get('prerequisites', []),
            estimated_duration_minutes=data.get('estimated_duration_minutes', 30),
            content_type=data.get('content_type', 'text')
        )
    
    def load_lesson(self, module_id: str, lesson_id: str) -> Optional[Lesson]:
        """Load a specific lesson."""
        module = self.load_module(module_id)
        if not module:
            return None
        
        return module.get_lesson_by_id(lesson_id)
    
    def reload_module(self, module_id: str) -> Optional[CourseModule]:
        """Reload a module, clearing cache."""
        if module_id in self._module_cache:
            del self._module_cache[module_id]
        return self.load_module(module_id)
    
    def clear_cache(self) -> None:
        """Clear all cached modules."""
        self._module_cache.clear()
        logger.info("Cleared content cache")