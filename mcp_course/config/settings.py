"""Course configuration settings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import os
from pydantic import Field
from pydantic_settings import BaseSettings


class CourseSettings(BaseSettings):
    """Main course configuration settings."""
    
    # Course metadata
    course_name: str = Field(default="MCP Course with Ollama", description="Course display name")
    course_version: str = Field(default="0.1.0", description="Course version")
    
    # File paths
    course_data_dir: Path = Field(default_factory=lambda: Path.home() / ".mcp_course", description="Course data directory")
    progress_file: str = Field(default="progress.json", description="Progress tracking file")
    
    # Learning settings
    auto_save_progress: bool = Field(default=True, description="Automatically save learning progress")
    show_hints: bool = Field(default=True, description="Show hints during exercises")
    difficulty_level: str = Field(default="beginner", description="Course difficulty level")
    
    class Config:
        env_prefix = "MCP_COURSE_"
        case_sensitive = False


@dataclass
class OllamaConfig:
    """Ollama integration configuration."""
    
    # Connection settings
    endpoint: str = "http://localhost:11434"
    timeout: int = 30
    max_retries: int = 3
    
    # Model settings
    default_model: str = "llama3.2:3b"
    available_models: list[str] = field(default_factory=lambda: [
        "llama3.2:3b",
        "llama3.2:1b", 
        "codellama:7b",
        "mistral:7b"
    ])
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    context_window: int = 4096
    
    # Privacy settings
    local_only: bool = True
    data_retention: str = "none"
    telemetry_enabled: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("Max tokens must be positive")
        if self.context_window < self.max_tokens:
            raise ValueError("Context window must be >= max tokens")


@dataclass
class ModuleConfig:
    """Configuration for individual course modules."""
    
    module_id: str
    name: str
    description: str
    prerequisites: list[str] = field(default_factory=list)
    estimated_duration: int = 60  # minutes
    difficulty: str = "beginner"
    interactive: bool = True
    
    def __post_init__(self):
        """Validate module configuration."""
        valid_difficulties = ["beginner", "intermediate", "advanced"]
        if self.difficulty not in valid_difficulties:
            raise ValueError(f"Difficulty must be one of {valid_difficulties}")


class ConfigManager:
    """Manages course configuration and settings."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or Path.home() / ".mcp_course" / "config.json"
        self._course_settings: Optional[CourseSettings] = None
        self._ollama_config: Optional[OllamaConfig] = None
        
    @property
    def course_settings(self) -> CourseSettings:
        """Get course settings, loading from environment if needed."""
        if self._course_settings is None:
            self._course_settings = CourseSettings()
        return self._course_settings
    
    @property
    def ollama_config(self) -> OllamaConfig:
        """Get Ollama configuration."""
        if self._ollama_config is None:
            self._ollama_config = OllamaConfig()
        return self._ollama_config
    
    def update_ollama_config(self, **kwargs) -> None:
        """Update Ollama configuration with new values."""
        if self._ollama_config is None:
            self._ollama_config = OllamaConfig()
        
        for key, value in kwargs.items():
            if hasattr(self._ollama_config, key):
                setattr(self._ollama_config, key, value)
            else:
                raise ValueError(f"Unknown Ollama config parameter: {key}")
    
    def get_module_config(self, module_id: str) -> Optional[ModuleConfig]:
        """Get configuration for a specific module."""
        # This would typically load from a configuration file
        # For now, return a default configuration
        module_configs = {
            "mcp_fundamentals": ModuleConfig(
                module_id="mcp_fundamentals",
                name="MCP Fundamentals",
                description="Learn the core concepts of Model Context Protocol",
                estimated_duration=90,
                difficulty="beginner"
            ),
            "ollama_setup": ModuleConfig(
                module_id="ollama_setup", 
                name="Ollama Setup & Configuration",
                description="Set up Ollama for local LLM hosting",
                prerequisites=["mcp_fundamentals"],
                estimated_duration=45,
                difficulty="beginner"
            ),
            "mcp_server": ModuleConfig(
                module_id="mcp_server",
                name="MCP Server Development", 
                description="Build your own MCP Servers",
                prerequisites=["mcp_fundamentals", "ollama_setup"],
                estimated_duration=120,
                difficulty="intermediate"
            )
        }
        
        return module_configs.get(module_id)
    
    def ensure_data_directory(self) -> Path:
        """Ensure course data directory exists."""
        data_dir = self.course_settings.course_data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir


# Global configuration instance
config_manager = ConfigManager()