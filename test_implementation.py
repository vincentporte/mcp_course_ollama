#!/usr/bin/env python3
"""
Test script to verify the course content management system implementation.
This script tests all the components we just implemented.
"""

import tempfile
import json
from pathlib import Path
from datetime import datetime

# Test imports
def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing imports...")
    
    try:
        from mcp_course.models import (
            CourseProgress, ExerciseCompletion, CourseModule, 
            Lesson, Exercise, LearningObjective, ContentLoader
        )
        from mcp_course.storage import ProgressStore
        from mcp_course.content import FileSystemContentLoader, CourseNavigator
        from mcp_course.utils.progress_utils import (
            serialize_progress_to_json, deserialize_progress_from_json,
            merge_progress_data, calculate_overall_progress
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_progress_models():
    """Test progress tracking models."""
    print("\n🔍 Testing progress models...")
    
    try:
        from mcp_course.models import CourseProgress, ExerciseCompletion
        
        # Test ExerciseCompletion
        exercise = ExerciseCompletion(
            exercise_id="ex_001",
            completed=True,
            code_submission="print('Hello, MCP!')",
            feedback="Great work!",
            completion_time=datetime.now(),
            attempts=2
        )
        
        # Test serialization
        exercise_dict = exercise.to_dict()
        exercise_restored = ExerciseCompletion.from_dict(exercise_dict)
        assert exercise.exercise_id == exercise_restored.exercise_id
        assert exercise.completed == exercise_restored.completed
        
        # Test CourseProgress
        progress = CourseProgress(
            user_id="test_user",
            module_id="module_001",
            completion_status="in_progress",
            assessment_score=85,
            practical_exercises=[exercise]
        )
        
        # Test progress methods
        progress.add_exercise_completion(exercise)
        retrieved_exercise = progress.get_exercise_completion("ex_001")
        assert retrieved_exercise is not None
        assert retrieved_exercise.completed
        
        completion_pct = progress.calculate_completion_percentage()
        assert completion_pct == 100.0
        
        # Test serialization
        progress_dict = progress.to_dict()
        progress_restored = CourseProgress.from_dict(progress_dict)
        assert progress.user_id == progress_restored.user_id
        assert progress.module_id == progress_restored.module_id
        
        print("✅ Progress models working correctly")
        return True
    except Exception as e:
        print(f"❌ Progress models test failed: {e}")
        return False


def test_content_models():
    """Test content structure models."""
    print("\n🔍 Testing content models...")
    
    try:
        from mcp_course.models import CourseModule, Lesson, Exercise, LearningObjective
        
        # Test LearningObjective
        objective = LearningObjective(
            id="obj_001",
            description="Understand MCP basics",
            level="beginner",
            skills=["mcp", "protocol"]
        )
        
        # Test Exercise
        exercise = Exercise(
            id="ex_001",
            title="Hello MCP",
            description="Create your first MCP server",
            difficulty="easy",
            estimated_minutes=15,
            code_template="# Your code here",
            hints=["Start with imports", "Define server class"]
        )
        
        # Test Lesson
        lesson = Lesson(
            id="lesson_001",
            title="MCP Introduction",
            description="Learn the basics of MCP",
            order=1,
            learning_objectives=[objective],
            exercises=[exercise],
            estimated_duration_minutes=30
        )
        
        # Test lesson methods
        retrieved_exercise = lesson.get_exercise_by_id("ex_001")
        assert retrieved_exercise is not None
        assert retrieved_exercise.title == "Hello MCP"
        
        # Test CourseModule
        module = CourseModule(
            id="module_001",
            title="MCP Fundamentals",
            description="Introduction to Model Context Protocol",
            order=1,
            lessons=[lesson],
            learning_objectives=[objective],
            difficulty_level="beginner"
        )
        
        # Test module methods
        module.add_lesson(lesson)
        retrieved_lesson = module.get_lesson_by_id("lesson_001")
        assert retrieved_lesson is not None
        
        next_lesson = module.get_next_lesson("lesson_001")
        assert next_lesson is None  # No next lesson
        
        all_exercises = module.get_all_exercises()
        assert len(all_exercises) == 1
        
        # Test serialization
        module_dict = module.to_dict()
        assert module_dict["id"] == "module_001"
        assert len(module_dict["lessons"]) == 1
        
        print("✅ Content models working correctly")
        return True
    except Exception as e:
        print(f"❌ Content models test failed: {e}")
        return False


def test_progress_store():
    """Test SQLite progress storage system."""
    print("\n🔍 Testing progress storage...")
    
    try:
        from mcp_course.storage import ProgressStore
        from mcp_course.models import CourseProgress, ExerciseCompletion
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ProgressStore(Path(temp_dir))
            
            # Create test progress with exercise
            exercise = ExerciseCompletion(
                exercise_id="ex_001",
                completed=True,
                code_submission="print('Hello SQLite!')",
                feedback="Great work!",
                completion_time=datetime.now(),
                attempts=1
            )
            
            progress = CourseProgress(
                user_id="test_user",
                module_id="module_001",
                completion_status="completed",
                assessment_score=90,
                practical_exercises=[exercise],
                notes="Test notes",
                time_spent_minutes=45
            )
            
            # Test save and load
            store.save_progress(progress)
            loaded_progress = store.load_progress("test_user", "module_001")
            
            assert loaded_progress is not None, "Failed to load saved progress"
            assert loaded_progress.user_id == "test_user"
            assert loaded_progress.module_id == "module_001"
            assert loaded_progress.assessment_score == 90
            assert loaded_progress.notes == "Test notes"
            assert loaded_progress.time_spent_minutes == 45
            assert len(loaded_progress.practical_exercises) == 1
            
            # Test exercise data
            loaded_exercise = loaded_progress.practical_exercises[0]
            assert loaded_exercise.exercise_id == "ex_001"
            assert loaded_exercise.completed == True
            assert loaded_exercise.code_submission == "print('Hello SQLite!')"
            assert loaded_exercise.feedback == "Great work!"
            assert loaded_exercise.attempts == 1
            
            # Test user progress loading
            user_progress = store.load_user_progress("test_user")
            assert "module_001" in user_progress
            assert len(user_progress) == 1
            
            # Test getting all users
            all_users = store.get_all_users()
            assert "test_user" in all_users
            
            # Test deletion of specific module
            deleted = store.delete_progress("test_user", "module_001")
            assert deleted
            
            # Verify deletion
            deleted_progress = store.load_progress("test_user", "module_001")
            assert deleted_progress is None
            
            # Test backup functionality
            backup_file = store.create_backup()
            assert backup_file is not None
            assert backup_file.exists()
        
        print("✅ Progress storage working correctly")
        return True
    except Exception as e:
        print(f"❌ Progress storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_content_loader():
    """Test content loading system."""
    print("\n🔍 Testing content loader...")
    
    try:
        from mcp_course.content import FileSystemContentLoader
        
        # Create temporary content structure
        with tempfile.TemporaryDirectory() as temp_dir:
            content_root = Path(temp_dir)
            
            # Create module directory
            module_dir = content_root / "module_001"
            module_dir.mkdir()
            
            # Create module configuration
            module_config = {
                "title": "Test Module",
                "description": "A test module",
                "order": 1,
                "difficulty_level": "beginner",
                "learning_objectives": [
                    {
                        "id": "obj_001",
                        "description": "Test objective",
                        "level": "beginner"
                    }
                ]
            }
            
            with open(module_dir / "module.json", "w") as f:
                json.dump(module_config, f)
            
            # Create lessons directory
            lessons_dir = module_dir / "lessons"
            lessons_dir.mkdir()
            
            # Create lesson configuration
            lesson_config = {
                "title": "Test Lesson",
                "description": "A test lesson",
                "order": 1,
                "exercises": [
                    {
                        "id": "ex_001",
                        "title": "Test Exercise",
                        "description": "A test exercise",
                        "difficulty": "easy",
                        "estimated_minutes": 10
                    }
                ]
            }
            
            with open(lessons_dir / "lesson_001.json", "w") as f:
                json.dump(lesson_config, f)
            
            # Test loader
            loader = FileSystemContentLoader(content_root)
            
            # Test getting available modules
            modules = loader.get_available_modules()
            assert "module_001" in modules
            
            # Test loading module
            module = loader.load_module("module_001")
            assert module is not None
            assert module.title == "Test Module"
            assert len(module.lessons) == 1
            
            # Test loading specific lesson
            lesson = loader.load_lesson("module_001", "lesson_001")
            assert lesson is not None
            assert lesson.title == "Test Lesson"
            assert len(lesson.exercises) == 1
            
            # Test cache
            module2 = loader.load_module("module_001")  # Should use cache
            assert module2 is module
            
            # Test reload
            module3 = loader.reload_module("module_001")
            assert module3 is not module  # Should be new instance
        
        print("✅ Content loader working correctly")
        return True
    except Exception as e:
        print(f"❌ Content loader test failed: {e}")
        return False


def test_navigation():
    """Test navigation system."""
    print("\n🔍 Testing navigation system...")
    
    try:
        from mcp_course.content import CourseNavigator
        from mcp_course.models import CourseProgress, ExerciseCompletion
        
        # Create mock content loader
        class MockContentLoader:
            def get_available_modules(self):
                return ["module_001", "module_002"]
            
            def load_module(self, module_id):
                from mcp_course.models import CourseModule, Lesson, Exercise
                
                if module_id == "module_001":
                    lesson1 = Lesson(
                        id="lesson_001", title="Lesson 1", description="First lesson",
                        order=1, exercises=[
                            Exercise(id="ex_001", title="Ex 1", description="Exercise 1", 
                                   difficulty="easy", estimated_minutes=10)
                        ]
                    )
                    lesson2 = Lesson(
                        id="lesson_002", title="Lesson 2", description="Second lesson",
                        order=2, exercises=[]
                    )
                    return CourseModule(
                        id="module_001", title="Module 1", description="First module",
                        order=1, lessons=[lesson1, lesson2]
                    )
                elif module_id == "module_002":
                    lesson3 = Lesson(
                        id="lesson_003", title="Lesson 3", description="Third lesson",
                        order=1, exercises=[]
                    )
                    return CourseModule(
                        id="module_002", title="Module 2", description="Second module",
                        order=2, lessons=[lesson3], prerequisites=["module_001"]
                    )
                return None
        
        navigator = CourseNavigator(MockContentLoader())
        
        # Test navigation context
        user_progress = {
            "module_001": CourseProgress(
                user_id="test_user",
                module_id="module_001",
                completion_status="in_progress",
                practical_exercises=[
                    ExerciseCompletion(exercise_id="ex_001", completed=False)
                ]
            )
        }
        
        context = navigator.get_navigation_context(user_progress)
        assert context.current_module_id == "module_001"
        assert context.current_lesson_id == "lesson_001"
        
        # Test next lesson navigation
        next_result = navigator.navigate_to_next_lesson(context)
        assert next_result.success
        assert next_result.target_lesson_id == "lesson_002"
        
        # Test progress summary
        summary = navigator.get_progress_summary(context)
        assert summary["total_modules"] == 2
        assert summary["completed_modules"] == 0
        
        print("✅ Navigation system working correctly")
        return True
    except Exception as e:
        print(f"❌ Navigation test failed: {e}")
        return False


def test_progress_utilities():
    """Test progress utility functions."""
    print("\n🔍 Testing progress utilities...")
    
    try:
        from mcp_course.utils.progress_utils import (
            serialize_progress_to_json, deserialize_progress_from_json,
            merge_progress_data, calculate_overall_progress
        )
        from mcp_course.models import CourseProgress, ExerciseCompletion
        
        # Create test progress
        exercise = ExerciseCompletion(
            exercise_id="ex_001",
            completed=True,
            code_submission="test code"
        )
        
        progress = CourseProgress(
            user_id="test_user",
            module_id="module_001",
            completion_status="completed",
            assessment_score=95,
            practical_exercises=[exercise]
        )
        
        # Test serialization
        json_str = serialize_progress_to_json(progress)
        assert isinstance(json_str, str)
        assert "test_user" in json_str
        
        # Test deserialization
        restored_progress = deserialize_progress_from_json(json_str)
        assert restored_progress.user_id == "test_user"
        assert restored_progress.assessment_score == 95
        
        # Test merging
        progress2 = CourseProgress(
            user_id="test_user",
            module_id="module_001",
            completion_status="completed",
            assessment_score=90,
            time_spent_minutes=120
        )
        
        merged = merge_progress_data(progress, progress2)
        assert merged.user_id == "test_user"
        assert merged.assessment_score in [90, 95]  # One of the two
        
        # Test overall progress calculation
        user_progress = {
            "module_001": progress,
            "module_002": CourseProgress(
                user_id="test_user",
                module_id="module_002",
                completion_status="in_progress"
            )
        }
        
        overall = calculate_overall_progress(user_progress)
        assert overall["total_modules"] == 2
        assert overall["completed_modules"] == 1
        assert overall["overall_completion_percentage"] == 50.0
        
        print("✅ Progress utilities working correctly")
        return True
    except Exception as e:
        print(f"❌ Progress utilities test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing MCP Course Content Management System Implementation")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_progress_models,
        test_content_models,
        test_progress_store,
        test_content_loader,
        test_navigation,
        test_progress_utilities
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)