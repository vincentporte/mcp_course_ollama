"""SQLite-based storage system for course progress tracking."""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from ..models.progress import CourseProgress, ExerciseCompletion
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ProgressStore:
    """Manages SQLite storage and retrieval of course progress data."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize progress store with storage directory."""
        if storage_dir is None:
            # Default to user's home directory
            storage_dir = Path.home() / ".mcp_course"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite database file
        self.db_path = self.storage_dir / "progress.db"
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized progress store at {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS course_progress (
                    user_id TEXT NOT NULL,
                    module_id TEXT NOT NULL,
                    completion_status TEXT NOT NULL,
                    assessment_score INTEGER,
                    last_accessed TEXT NOT NULL,
                    notes TEXT,
                    time_spent_minutes INTEGER DEFAULT 0,
                    PRIMARY KEY (user_id, module_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exercise_completions (
                    user_id TEXT NOT NULL,
                    module_id TEXT NOT NULL,
                    exercise_id TEXT NOT NULL,
                    completed BOOLEAN NOT NULL,
                    code_submission TEXT,
                    feedback TEXT,
                    completion_time TEXT,
                    attempts INTEGER DEFAULT 0,
                    PRIMARY KEY (user_id, module_id, exercise_id),
                    FOREIGN KEY (user_id, module_id) REFERENCES course_progress(user_id, module_id)
                )
            """)
            
            conn.commit()
    
    def save_progress(self, progress: CourseProgress) -> None:
        """Save course progress for a user and module."""
        with sqlite3.connect(self.db_path) as conn:
            # Save main progress record
            conn.execute("""
                INSERT OR REPLACE INTO course_progress 
                (user_id, module_id, completion_status, assessment_score, 
                 last_accessed, notes, time_spent_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                progress.user_id,
                progress.module_id,
                progress.completion_status,
                progress.assessment_score,
                progress.last_accessed.isoformat(),
                progress.notes,
                progress.time_spent_minutes
            ))
            
            # Delete existing exercise completions for this module
            conn.execute("""
                DELETE FROM exercise_completions 
                WHERE user_id = ? AND module_id = ?
            """, (progress.user_id, progress.module_id))
            
            # Save exercise completions
            for exercise in progress.practical_exercises:
                conn.execute("""
                    INSERT INTO exercise_completions
                    (user_id, module_id, exercise_id, completed, code_submission,
                     feedback, completion_time, attempts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    progress.user_id,
                    progress.module_id,
                    exercise.exercise_id,
                    exercise.completed,
                    exercise.code_submission,
                    exercise.feedback,
                    exercise.completion_time.isoformat() if exercise.completion_time else None,
                    exercise.attempts
                ))
            
            conn.commit()
        
        logger.info(f"Saved progress for user {progress.user_id}, module {progress.module_id}")
    
    def load_progress(self, user_id: str, module_id: str) -> Optional[CourseProgress]:
        """Load course progress for a specific user and module."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Load main progress record
            cursor = conn.execute("""
                SELECT * FROM course_progress 
                WHERE user_id = ? AND module_id = ?
            """, (user_id, module_id))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Load exercise completions
            cursor = conn.execute("""
                SELECT * FROM exercise_completions 
                WHERE user_id = ? AND module_id = ?
            """, (user_id, module_id))
            
            exercises = []
            for ex_row in cursor.fetchall():
                completion_time = None
                if ex_row['completion_time']:
                    completion_time = datetime.fromisoformat(ex_row['completion_time'])
                
                exercises.append(ExerciseCompletion(
                    exercise_id=ex_row['exercise_id'],
                    completed=bool(ex_row['completed']),
                    code_submission=ex_row['code_submission'],
                    feedback=ex_row['feedback'],
                    completion_time=completion_time,
                    attempts=ex_row['attempts']
                ))
            
            return CourseProgress(
                user_id=row['user_id'],
                module_id=row['module_id'],
                completion_status=row['completion_status'],
                assessment_score=row['assessment_score'],
                last_accessed=datetime.fromisoformat(row['last_accessed']),
                practical_exercises=exercises,
                notes=row['notes'],
                time_spent_minutes=row['time_spent_minutes']
            )
    
    def load_user_progress(self, user_id: str) -> Dict[str, CourseProgress]:
        """Load all progress for a specific user."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT module_id FROM course_progress WHERE user_id = ?
            """, (user_id,))
            
            progress = {}
            for row in cursor.fetchall():
                module_progress = self.load_progress(user_id, row['module_id'])
                if module_progress:
                    progress[row['module_id']] = module_progress
            
            return progress
    
    def get_all_users(self) -> List[str]:
        """Get list of all users with stored progress."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT user_id FROM course_progress")
            return [row[0] for row in cursor.fetchall()]
    
    def delete_progress(self, user_id: str, module_id: Optional[str] = None) -> bool:
        """Delete progress data for user and optionally specific module."""
        with sqlite3.connect(self.db_path) as conn:
            if module_id is None:
                # Delete all progress for user
                cursor = conn.execute("""
                    DELETE FROM course_progress WHERE user_id = ?
                """, (user_id,))
                
                conn.execute("""
                    DELETE FROM exercise_completions WHERE user_id = ?
                """, (user_id,))
                
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Deleted all progress for user {user_id}")
            else:
                # Delete specific module progress
                cursor = conn.execute("""
                    DELETE FROM course_progress WHERE user_id = ? AND module_id = ?
                """, (user_id, module_id))
                
                conn.execute("""
                    DELETE FROM exercise_completions WHERE user_id = ? AND module_id = ?
                """, (user_id, module_id))
                
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Deleted progress for user {user_id}, module {module_id}")
            
            conn.commit()
            return deleted
    
    def create_backup(self) -> Optional[Path]:
        """Create a backup of current progress data."""
        if not self.db_path.exists():
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.storage_dir / f"progress_backup_{timestamp}.db"
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_file)
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
            # Create backup of current data before restoring
            self.create_backup()
            
            # Close any existing connections and copy backup
            import shutil
            import time
            
            # Small delay to ensure connections are closed
            time.sleep(0.1)
            
            # Copy backup to main database file
            shutil.copy2(backup_file, self.db_path)
            
            # Reinitialize database to ensure proper connection
            self._init_database()
            
            logger.info(f"Restored progress from backup {backup_file}")
            return True
            
        except OSError as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False