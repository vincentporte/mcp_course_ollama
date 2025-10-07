"""Local storage system for course progress tracking."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from ..models.progress import CourseProgress, ExerciseCompletion
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ProgressStore:
    """Manages local storage and retrieval of course progress data."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize progress store with storage directory."""
        if storage_dir is None:
            # Default to user's home directory
            storage_dir = Path.home() / ".mcp_course" / "progress"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File for storing all progress data
        self.progress_file = self.storage_dir / "course_progress.json"
        
        logger.info(f"Initialized progress store at {self.storage_dir}")
    
    def _load_all_progress(self) -> Dict[str, Dict[str, CourseProgress]]:
        """Load all progress data from storage."""
        if not self.progress_file.exists():
            return {}
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert nested dictionaries back to CourseProgress objects
            progress_data = {}
            for user_id, user_modules in data.items():
                progress_data[user_id] = {}
                for module_id, module_data in user_modules.items():
                    progress_data[user_id][module_id] = CourseProgress.from_dict(module_data)
            
            return progress_data
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error loading progress data: {e}")
            # Backup corrupted file and start fresh
            backup_file = self.progress_file.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            if self.progress_file.exists():
                self.progress_file.rename(backup_file)
                logger.warning(f"Corrupted progress file backed up to {backup_file}")
            return {}
    
    def _save_all_progress(self, progress_data: Dict[str, Dict[str, CourseProgress]]) -> None:
        """Save all progress data to storage."""
        try:
            # Convert CourseProgress objects to dictionaries
            serializable_data = {}
            for user_id, user_modules in progress_data.items():
                serializable_data[user_id] = {}
                for module_id, progress in user_modules.items():
                    serializable_data[user_id][module_id] = progress.to_dict()
            
            # Write to temporary file first, then rename for atomic operation
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            temp_file.rename(self.progress_file)
            logger.debug("Progress data saved successfully")
            
        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Error saving progress data: {e}")
            raise
    
    def save_progress(self, progress: CourseProgress) -> None:
        """Save course progress for a user and module."""
        all_progress = self._load_all_progress()
        
        # Ensure user exists in data structure
        if progress.user_id not in all_progress:
            all_progress[progress.user_id] = {}
        
        # Update progress for this module
        all_progress[progress.user_id][progress.module_id] = progress
        
        self._save_all_progress(all_progress)
        logger.info(f"Saved progress for user {progress.user_id}, module {progress.module_id}")
    
    def load_progress(self, user_id: str, module_id: str) -> Optional[CourseProgress]:
        """Load course progress for a specific user and module."""
        all_progress = self._load_all_progress()
        
        user_progress = all_progress.get(user_id, {})
        return user_progress.get(module_id)
    
    def load_user_progress(self, user_id: str) -> Dict[str, CourseProgress]:
        """Load all progress for a specific user."""
        all_progress = self._load_all_progress()
        return all_progress.get(user_id, {})
    
    def get_all_users(self) -> List[str]:
        """Get list of all users with stored progress."""
        all_progress = self._load_all_progress()
        return list(all_progress.keys())
    
    def delete_progress(self, user_id: str, module_id: Optional[str] = None) -> bool:
        """Delete progress data for user and optionally specific module."""
        all_progress = self._load_all_progress()
        
        if user_id not in all_progress:
            return False
        
        if module_id is None:
            # Delete all progress for user
            del all_progress[user_id]
            logger.info(f"Deleted all progress for user {user_id}")
        else:
            # Delete specific module progress
            if module_id in all_progress[user_id]:
                del all_progress[user_id][module_id]
                logger.info(f"Deleted progress for user {user_id}, module {module_id}")
                
                # Remove user entry if no modules left
                if not all_progress[user_id]:
                    del all_progress[user_id]
            else:
                return False
        
        self._save_all_progress(all_progress)
        return True
    
    def create_backup(self) -> Optional[Path]:
        """Create a backup of current progress data."""
        if not self.progress_file.exists():
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.storage_dir / f"progress_backup_{timestamp}.json"
        
        try:
            import shutil
            shutil.copy2(self.progress_file, backup_file)
            logger.info(f"Created progress backup at {backup_file}")
            return backup_file
        except OSError as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def restore_from_backup(self, backup_file: Path) -> bool:
        """Restore progress data from a backup file."""
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        try:
            # Validate backup file by trying to load it
            with open(backup_file, 'r', encoding='utf-8') as f:
                json.load(f)
            
            # Create backup of current data before restoring
            self.create_backup()
            
            # Copy backup to main progress file
            import shutil
            shutil.copy2(backup_file, self.progress_file)
            
            logger.info(f"Restored progress from backup {backup_file}")
            return True
            
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False