"""Course configuration settings."""

from dataclasses import dataclass, field
import os
from pathlib import Path


MAX_TOKENS_MIN: int = 1
TEMPERATURE_MIN: float = 0.0
TEMPERATURE_MAX: float = 2.0
DURATION_MIN: int = 0
DURATION_MAX: int = 480


@dataclass
class CourseSettings:
    """Main course configuration settings."""

    # Course metadata
    course_name: str = "MCP Course with Ollama"
    course_version: str = "0.1.0"

    # File paths
    course_data_dir: Path = field(default_factory=lambda: Path.home() / ".mcp_course")

    # Learning settings
    show_hints: bool = True
    difficulty_level: str = "beginner"

    @classmethod
    def from_env(cls) -> "CourseSettings":
        """Create settings from environment variables."""
        return cls(
            course_name=os.getenv("MCP_COURSE_NAME", "MCP Course with Ollama"),
            course_version=os.getenv("MCP_COURSE_VERSION", "0.1.0"),
            course_data_dir=Path(os.getenv("MCP_COURSE_DATA_DIR", str(Path.home() / ".mcp_course"))),
            show_hints=os.getenv("MCP_COURSE_SHOW_HINTS", "true").lower() == "true",
            difficulty_level=os.getenv("MCP_COURSE_DIFFICULTY", "beginner")
        )


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
        if self.temperature < TEMPERATURE_MIN or self.temperature > TEMPERATURE_MAX:
            raise ValueError(f"Temperature must be between {TEMPERATURE_MIN} and {TEMPERATURE_MAX}")
        if self.max_tokens < MAX_TOKENS_MIN:
            raise ValueError("Max tokens must be positive")
        if self.context_window < self.max_tokens:
            raise ValueError("Context window must be >= max tokens")

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Create Ollama config from environment variables."""
        return cls(
            endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "30")),
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
            default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "2048")),
            context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "4096")),
            local_only=os.getenv("OLLAMA_LOCAL_ONLY", "true").lower() == "true",
            data_retention=os.getenv("OLLAMA_DATA_RETENTION", "none"),
            telemetry_enabled=os.getenv("OLLAMA_TELEMETRY", "false").lower() == "true"
        )


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
    def __init__(self, config_path: Path | None = None):
        """Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or Path.home() / ".mcp_course" / "config.json"
        self._course_settings: CourseSettings | None = None
        self._ollama_config: OllamaConfig | None = None

    @property
    def course_settings(self) -> CourseSettings:
        """Get course settings, loading from environment if needed."""
        if self._course_settings is None:
            self._course_settings = CourseSettings.from_env()
        return self._course_settings

    @property
    def ollama_config(self) -> OllamaConfig:
        """Get Ollama configuration."""
        if self._ollama_config is None:
            self._ollama_config = OllamaConfig.from_env()
        return self._ollama_config

    def update_ollama_config(self, **kwargs) -> None:
        """Update Ollama configuration with new values."""
        if self._ollama_config is None:
            self._ollama_config = OllamaConfig.from_env()

        for key, value in kwargs.items():
            if hasattr(self._ollama_config, key):
                setattr(self._ollama_config, key, value)
            else:
                raise ValueError(f"Unknown Ollama config parameter: {key}")

    def get_module_config(self, module_id: str) -> ModuleConfig | None:
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
