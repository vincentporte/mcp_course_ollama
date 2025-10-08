"""Ollama setup verification and installation utilities.

This module provides tools for verifying Ollama installation, guiding users
through setup, and ensuring proper configuration for the MCP course platform.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import httpx
import psutil

from mcp_course.ollama_client.config import OllamaConfig
from mcp_course.ollama_client.health import HealthCheckResult, OllamaHealthChecker


logger = logging.getLogger(__name__)

SMALL_MEMORY_AVAIL: int = 4
MEDIUM_MEMORY_AVAIL: int = 8
LARGE_MEMOMRY_AVAIL: int = 16

@dataclass
class SystemInfo:
    """System information for Ollama compatibility checking.

    Attributes:
        os_name: Operating system name
        os_version: Operating system version
        architecture: System architecture
        python_version: Python version
        available_memory: Available system memory in GB
        cpu_cores: Number of CPU cores
    """
    os_name: str
    os_version: str
    architecture: str
    python_version: str
    available_memory: float | None = None
    cpu_cores: int | None = None


@dataclass
class InstallationStatus:
    """Status of Ollama installation.

    Attributes:
        is_installed: Whether Ollama is installed
        version: Ollama version if installed
        executable_path: Path to Ollama executable
        service_running: Whether Ollama service is running
        installation_method: How Ollama was installed
        issues: List of detected issues
    """
    is_installed: bool
    version: str | None = None
    executable_path: str | None = None
    service_running: bool = False
    installation_method: str | None = None
    issues: list[str] | None = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class OllamaSetupVerifier:
    """Verifies and guides Ollama installation and setup.

    This class provides comprehensive setup verification, installation guidance,
    and troubleshooting for Ollama in the context of the MCP course.
    """

    def __init__(self):
        """Initialize the setup verifier."""
        self.system_info = self._gather_system_info()

    def _gather_system_info(self) -> SystemInfo:
        """Gather system information for compatibility checking.

        Returns:
            SystemInfo with current system details
        """
        try:
            # Get memory info (rough estimation)
            available_memory = None
            cpu_cores = None

            try:
                available_memory = psutil.virtual_memory().total / (1024**3)  # GB
                cpu_cores = psutil.cpu_count()
            except Exception as e:
                logger.warning(f"Failed to get system info with psutil: {e}")

            return SystemInfo(
                os_name=platform.system(),
                os_version=platform.version(),
                architecture=platform.machine(),
                python_version=sys.version,
                available_memory=available_memory,
                cpu_cores=cpu_cores
            )
        except Exception as e:
            logger.error(f"Failed to gather system info: {e}")
            return SystemInfo(
                os_name="Unknown",
                os_version="Unknown",
                architecture="Unknown",
                python_version=sys.version
            )

    def check_system_compatibility(self) -> HealthCheckResult:
        """Check if the system is compatible with Ollama.

        Returns:
            HealthCheckResult with compatibility status
        """
        issues = []
        warnings = []

        # Check operating system
        supported_os = ["Linux", "Darwin", "Windows"]
        if self.system_info.os_name not in supported_os:
            issues.append(f"Unsupported operating system: {self.system_info.os_name}")

        # Check architecture
        supported_archs = ["x86_64", "AMD64", "arm64", "aarch64"]
        if self.system_info.architecture not in supported_archs:
            warnings.append(f"Architecture {self.system_info.architecture} may not be fully supported")

        # Check memory (if available)
        if self.system_info.available_memory:
            if self.system_info.available_memory < SMALL_MEMORY_AVAIL:
                warnings.append("Less than 4GB RAM available - may impact model performance")
            elif self.system_info.available_memory < MEDIUM_MEMORY_AVAIL:
                warnings.append("Less than 8GB RAM available - larger models may not run well")

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            issues.append(f"Python {python_version.major}.{python_version.minor} is too old, need 3.8+")

        is_compatible = len(issues) == 0
        status = "System is compatible with Ollama" if is_compatible else "System compatibility issues found"

        return HealthCheckResult(
            is_healthy=is_compatible,
            status=status,
            details={
                "system_info": self.system_info.__dict__,
                "issues": issues,
                "warnings": warnings
            }
        )

    def check_ollama_installation(self) -> InstallationStatus:
        """Check current Ollama installation status.

        Returns:
            InstallationStatus with detailed installation information
        """
        status = InstallationStatus(is_installed=False)

        # Check if ollama executable exists
        ollama_path = shutil.which("ollama")
        if ollama_path:
            status.is_installed = True
            status.executable_path = ollama_path

            # Get version
            try:
                result = subprocess.run(
                    ["ollama", "--version"],
                    check=False, capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    status.version = result.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                status.issues.append(f"Failed to get Ollama version: {e}")

            # Check if service is running
            status.service_running = self._is_ollama_service_running()

            # Detect installation method
            status.installation_method = self._detect_installation_method(ollama_path)
        else:
            status.issues.append("Ollama executable not found in PATH")

        return status

    def _is_ollama_service_running(self) -> bool:
        """Check if Ollama service is running.

        Returns:
            True if service is running
        """
        try:
            # Try to connect to default Ollama endpoint
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            return response.is_success
        except Exception:
            return False

    def _detect_installation_method(self, executable_path: str) -> str:
        """Detect how Ollama was installed.

        Args:
            executable_path: Path to Ollama executable

        Returns:
            Installation method description
        """
        path = Path(executable_path)

        # Check common installation paths
        if "/usr/local/bin" in str(path):
            return "System installation (likely curl script)"
        elif "/.local/bin" in str(path):
            return "User installation"
        elif "/opt/" in str(path):
            return "Package manager installation"
        elif "homebrew" in str(path).lower() or "/brew/" in str(path):
            return "Homebrew installation"
        elif "conda" in str(path).lower():
            return "Conda installation"
        else:
            return "Unknown installation method"

    def get_installation_instructions(self) -> dict[str, str]:
        """Get platform-specific installation instructions.

        Returns:
            Dictionary with installation instructions for current platform
        """
        os_name = self.system_info.os_name.lower()

        instructions = {
            "linux": {
                "title": "Install Ollama on Linux",
                "method": "curl",
                "command": "curl -fsSL https://ollama.ai/install.sh | sh",
                "description": "Download and run the official Ollama installation script",
                "alternative": "You can also download from https://ollama.ai/download/linux",
                "post_install": [
                    "Start Ollama service: ollama serve",
                    "In another terminal, pull a model: ollama pull llama3.2:3b"
                ]
            },
            "darwin": {
                "title": "Install Ollama on macOS",
                "method": "download",
                "command": "Download from https://ollama.ai/download/mac",
                "description": "Download the .dmg file and install like any other Mac application",
                "alternative": "Or use Homebrew: brew install ollama",
                "post_install": [
                    "Ollama will start automatically after installation",
                    "Pull a model: ollama pull llama3.2:3b"
                ]
            },
            "windows": {
                "title": "Install Ollama on Windows",
                "method": "download",
                "command": "Download from https://ollama.ai/download/windows",
                "description": "Download the .exe installer and run it",
                "alternative": "Ollama will be available in the system tray after installation",
                "post_install": [
                    "Ollama starts automatically as a Windows service",
                    "Open Command Prompt and run: ollama pull llama3.2:3b"
                ]
            }
        }

        return instructions.get(os_name, instructions["linux"])

    def verify_installation_complete(self) -> HealthCheckResult:
        """Verify that Ollama installation is complete and working.

        Returns:
            HealthCheckResult with verification status
        """
        # Check installation status
        install_status = self.check_ollama_installation()

        if not install_status.is_installed:
            return HealthCheckResult(
                is_healthy=False,
                status="Ollama is not installed",
                details={"installation_status": install_status.__dict__}
            )

        if not install_status.service_running:
            return HealthCheckResult(
                is_healthy=False,
                status="Ollama is installed but service is not running",
                details={
                    "installation_status": install_status.__dict__,
                    "suggestion": "Try running 'ollama serve' in a terminal"
                }
            )

        # Test basic functionality with health checker
        try:
            config = OllamaConfig.create_default("llama3.2:3b")  # Default test model
            health_checker = OllamaHealthChecker(config)
            connection_result = health_checker.check_connection()

            if connection_result.is_healthy:
                return HealthCheckResult(
                    is_healthy=True,
                    status="Ollama installation is complete and working",
                    details={
                        "installation_status": install_status.__dict__,
                        "connection_test": connection_result.__dict__
                    }
                )
            else:
                return HealthCheckResult(
                    is_healthy=False,
                    status="Ollama is installed but connection test failed",
                    details={
                        "installation_status": install_status.__dict__,
                        "connection_test": connection_result.__dict__
                    }
                )
        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                status=f"Ollama verification failed: {e}",
                details={"installation_status": install_status.__dict__, "error": str(e)}
            )

    def get_setup_recommendations(self) -> dict[str, list[str]]:
        """Get setup recommendations based on system analysis.

        Returns:
            Dictionary with categorized recommendations
        """
        recommendations = {
            "required": [],
            "recommended": [],
            "optional": []
        }

        # Check system compatibility
        compat_result = self.check_system_compatibility()
        if not compat_result.is_healthy:
            recommendations["required"].extend([
                f"Address system compatibility issues: {issue}"
                for issue in compat_result.details.get("issues", [])
            ])

        # Check installation
        install_status = self.check_ollama_installation()
        if not install_status.is_installed:
            instructions = self.get_installation_instructions()
            recommendations["required"].append(f"Install Ollama: {instructions['command']}")
        elif not install_status.service_running:
            recommendations["required"].append("Start Ollama service: ollama serve")

        # Memory recommendations
        if self.system_info.available_memory:
            if self.system_info.available_memory < MEDIUM_MEMORY_AVAIL:
                recommendations["recommended"].append(
                    "Consider using smaller models (e.g., llama3.2:1b) for better performance"
                )
            elif self.system_info.available_memory >= LARGE_MEMOMRY_AVAIL:
                recommendations["optional"].append(
                    "You have sufficient memory for larger models (e.g., llama3.2:8b (?), llama3.2:70b (?))"
                )

        # Course-specific recommendations
        recommendations["recommended"].extend([
            "Pull the recommended course model: ollama pull llama3.2:3b",
            "Test model functionality: ollama run llama3.2:3b 'Hello, world!'",
            "Consider pulling a smaller model for faster responses: ollama pull llama3.2:1b"
        ])

        recommendations["optional"].extend([
            "Install additional models for comparison: ollama pull mistral",
            "Set up Ollama to start automatically on system boot",
            "Configure Ollama with custom model parameters"
        ])

        return recommendations

    def run_complete_setup_check(self) -> dict[str, HealthCheckResult]:
        """Run a complete setup verification check.

        Returns:
            Dictionary of check results
        """
        results = {}

        logger.info("Running system compatibility check...")
        results["system_compatibility"] = self.check_system_compatibility()

        logger.info("Running installation verification...")
        results["installation_verification"] = self.verify_installation_complete()

        return results
