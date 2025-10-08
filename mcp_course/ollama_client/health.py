"""Ollama health check and connection testing utilities.

This module provides comprehensive health checking capabilities for Ollama
servers, including connection testing, model availability, and performance
monitoring for the MCP course platform.
"""

from dataclasses import dataclass
import logging
import time

from mcp_course.ollama_client.client import OllamaClient, OllamaConnectionError
from mcp_course.ollama_client.config import OllamaConfig


logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check operation.

    Attributes:
        is_healthy: Whether the check passed
        status: Status message
        details: Additional details about the check
        response_time: Time taken for the check in seconds
        timestamp: When the check was performed
    """
    is_healthy: bool
    status: str
    details: dict[str, any] = None
    response_time: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model.

    Attributes:
        model_name: Name of the model
        avg_response_time: Average response time in seconds
        tokens_per_second: Tokens generated per second
        memory_usage: Estimated memory usage in MB
        test_prompt_length: Length of test prompt used
        test_response_length: Length of test response
    """
    model_name: str
    avg_response_time: float
    tokens_per_second: float
    memory_usage: float | None = None
    test_prompt_length: int = 0
    test_response_length: int = 0


class OllamaHealthChecker:
    """Comprehensive health checker for Ollama servers.

    This class provides various health check capabilities including
    connection testing, model availability checks, and performance monitoring.
    """

    def __init__(self, config: OllamaConfig):
        """Initialize the health checker.

        Args:
            config: Ollama configuration to use for health checks
        """
        self.config = config
        self.client = OllamaClient(config)

    def check_connection(self) -> HealthCheckResult:
        """Check basic connection to Ollama server.

        Returns:
            HealthCheckResult with connection status
        """
        start_time = time.time()

        try:
            # Try to list models as a basic connectivity test
            models = self.client.list_models()
            response_time = time.time() - start_time

            # Safely extract model names
            model_names = []
            for model in models[:5]:  # First 5 models
                if isinstance(model, dict) and 'name' in model:
                    model_names.append(model['name'])
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
                else:
                    model_names.append(str(model))

            return HealthCheckResult(
                is_healthy=True,
                status="Connection successful",
                details={
                    "endpoint": self.config.endpoint,
                    "models_count": len(models),
                    "models": model_names
                },
                response_time=response_time
            )

        except OllamaConnectionError as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                is_healthy=False,
                status=f"Connection failed: {e}",
                details={"endpoint": self.config.endpoint, "error_type": "connection"},
                response_time=response_time
            )
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                is_healthy=False,
                status=f"Unexpected error: {e}",
                details={"endpoint": self.config.endpoint, "error_type": "unknown"},
                response_time=response_time
            )

    async def check_connection_async(self) -> HealthCheckResult:
        """Check basic connection to Ollama server asynchronously.

        Returns:
            HealthCheckResult with connection status
        """
        start_time = time.time()

        try:
            # Try to list models as a basic connectivity test
            models = await self.client.list_models_async()
            response_time = time.time() - start_time

            # Safely extract model names
            model_names = []
            for model in models[:5]:  # First 5 models
                if isinstance(model, dict) and 'name' in model:
                    model_names.append(model['name'])
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
                else:
                    model_names.append(str(model))

            return HealthCheckResult(
                is_healthy=True,
                status="Connection successful",
                details={
                    "endpoint": self.config.endpoint,
                    "models_count": len(models),
                    "models": model_names
                },
                response_time=response_time
            )

        except OllamaConnectionError as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                is_healthy=False,
                status=f"Connection failed: {e}",
                details={"endpoint": self.config.endpoint, "error_type": "connection"},
                response_time=response_time
            )
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                is_healthy=False,
                status=f"Unexpected error: {e}",
                details={"endpoint": self.config.endpoint, "error_type": "unknown"},
                response_time=response_time
            )

    def check_model_availability(self, model_name: str | None = None) -> HealthCheckResult:
        """Check if a specific model is available.

        Args:
            model_name: Model to check (defaults to config model)

        Returns:
            HealthCheckResult with model availability status
        """
        target_model = model_name or self.config.model_name
        start_time = time.time()

        try:
            is_available = self.client.is_model_available(target_model)
            response_time = time.time() - start_time

            if is_available:
                model_info = self.client.get_model_info(target_model)
                return HealthCheckResult(
                    is_healthy=True,
                    status=f"Model '{target_model}' is available",
                    details={
                        "model_name": target_model,
                        "model_info": model_info
                    },
                    response_time=response_time
                )
            else:
                return HealthCheckResult(
                    is_healthy=False,
                    status=f"Model '{target_model}' is not available",
                    details={"model_name": target_model},
                    response_time=response_time
                )

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                is_healthy=False,
                status=f"Model check failed: {e}",
                details={"model_name": target_model, "error": str(e)},
                response_time=response_time
            )

    async def check_model_availability_async(self, model_name: str | None = None) -> HealthCheckResult:
        """Check if a specific model is available asynchronously.

        Args:
            model_name: Model to check (defaults to config model)

        Returns:
            HealthCheckResult with model availability status
        """
        target_model = model_name or self.config.model_name
        start_time = time.time()

        try:
            is_available = await self.client.is_model_available_async(target_model)
            response_time = time.time() - start_time

            if is_available:
                model_info = await self.client.get_model_info_async(target_model)
                return HealthCheckResult(
                    is_healthy=True,
                    status=f"Model '{target_model}' is available",
                    details={
                        "model_name": target_model,
                        "model_info": model_info
                    },
                    response_time=response_time
                )
            else:
                return HealthCheckResult(
                    is_healthy=False,
                    status=f"Model '{target_model}' is not available",
                    details={"model_name": target_model},
                    response_time=response_time
                )

        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                is_healthy=False,
                status=f"Model check failed: {e}",
                details={"model_name": target_model, "error": str(e)},
                response_time=response_time
            )

    def test_model_generation(
        self,
        test_prompt: str = "Hello, how are you?",
        model_name: str | None = None,
        preview_length: int = 100,
    ) -> HealthCheckResult:
        """Test model generation capability.

        Args:
            test_prompt: Prompt to use for testing
            model_name: Model to test (defaults to config model)

        Returns:
            HealthCheckResult with generation test status
        """
        target_model = model_name or self.config.model_name
        start_time = time.time()

        try:
            # Temporarily use the test model
            original_model = self.config.model_name
            self.config.model_name = target_model

            response = self.client.generate(test_prompt)
            response_time = time.time() - start_time

            # Restore original model
            self.config.model_name = original_model

            if response and 'message' in response:
                content = response['message'].get('content', '')
                return HealthCheckResult(
                    is_healthy=True,
                    status="Model generation successful",
                    details={
                        "model_name": target_model,
                        "prompt_length": len(test_prompt),
                        "response_length": len(content),
                        "response_preview": content[:preview_length] + "..." if len(content) > preview_length else content
                    },
                    response_time=response_time
                )
            else:
                return HealthCheckResult(
                    is_healthy=False,
                    status="Model generation returned empty response",
                    details={"model_name": target_model},
                    response_time=response_time
                )

        except Exception as e:
            response_time = time.time() - start_time
            # Restore original model in case of error
            self.config.model_name = original_model
            return HealthCheckResult(
                is_healthy=False,
                status=f"Model generation failed: {e}",
                details={"model_name": target_model, "error": str(e)},
                response_time=response_time
            )

    async def test_model_generation_async(
        self,
        test_prompt: str = "Hello, how are you?",
        model_name: str | None = None,
        preview_length: int = 100,
    ) -> HealthCheckResult:
        """Test model generation capability asynchronously.

        Args:
            test_prompt: Prompt to use for testing
            model_name: Model to test (defaults to config model)

        Returns:
            HealthCheckResult with generation test status
        """
        target_model = model_name or self.config.model_name
        start_time = time.time()

        try:
            # Temporarily use the test model
            original_model = self.config.model_name
            self.config.model_name = target_model

            response = await self.client.generate_async(test_prompt)
            response_time = time.time() - start_time

            # Restore original model
            self.config.model_name = original_model

            if response and 'message' in response:
                content = response['message'].get('content', '')
                return HealthCheckResult(
                    is_healthy=True,
                    status="Model generation successful",
                    details={
                        "model_name": target_model,
                        "prompt_length": len(test_prompt),
                        "response_length": len(content),
                        "response_preview": content[:preview_length] + "..." if len(content) > preview_length else content
                    },
                    response_time=response_time
                )
            else:
                return HealthCheckResult(
                    is_healthy=False,
                    status="Model generation returned empty response",
                    details={"model_name": target_model},
                    response_time=response_time
                )

        except Exception as e:
            response_time = time.time() - start_time
            # Restore original model in case of error
            self.config.model_name = original_model
            return HealthCheckResult(
                is_healthy=False,
                status=f"Model generation failed: {e}",
                details={"model_name": target_model, "error": str(e)},
                response_time=response_time
            )

    def run_comprehensive_health_check(self) -> dict[str, HealthCheckResult]:
        """Run a comprehensive health check covering all aspects.

        Returns:
            Dictionary of check names to HealthCheckResult objects
        """
        results = {}

        # Connection check
        logger.info("Running connection health check...")
        results['connection'] = self.check_connection()

        # Model availability check
        if results['connection'].is_healthy:
            logger.info("Running model availability check...")
            results['model_availability'] = self.check_model_availability()

            # Generation test
            if results['model_availability'].is_healthy:
                logger.info("Running model generation test...")
                results['model_generation'] = self.test_model_generation()

        return results

    async def run_comprehensive_health_check_async(self) -> dict[str, HealthCheckResult]:
        """Run a comprehensive health check covering all aspects asynchronously.

        Returns:
            Dictionary of check names to HealthCheckResult objects
        """
        results = {}

        # Connection check
        logger.info("Running connection health check...")
        results['connection'] = await self.check_connection_async()

        # Model availability check
        if results['connection'].is_healthy:
            logger.info("Running model availability check...")
            results['model_availability'] = await self.check_model_availability_async()

            # Generation test
            if results['model_availability'].is_healthy:
                logger.info("Running model generation test...")
                results['model_generation'] = await self.test_model_generation_async()

        return results

    def measure_model_performance(
        self,
        test_prompts: list[str] | None = None,
        model_name: str | None = None,
        num_runs: int = 3
    ) -> ModelPerformanceMetrics:
        """Measure model performance metrics.

        Args:
            test_prompts: List of prompts to test with
            model_name: Model to test (defaults to config model)
            num_runs: Number of test runs for averaging

        Returns:
            ModelPerformanceMetrics with performance data
        """
        target_model = model_name or self.config.model_name

        if test_prompts is None:
            test_prompts = [
                "Explain what MCP (Model Context Protocol) is in one sentence.",
                "Write a simple Python function that adds two numbers.",
                "What are the benefits of using local LLMs?"
            ]

        total_response_time = 0.0
        total_tokens = 0
        total_prompt_length = 0
        total_response_length = 0
        successful_runs = 0

        # Temporarily use the test model
        original_model = self.config.model_name
        self.config.model_name = target_model

        try:
            for _ in range(num_runs):
                for prompt in test_prompts:
                    try:
                        start_time = time.time()
                        response = self.client.generate(prompt)
                        end_time = time.time()

                        if response and 'message' in response:
                            content = response['message'].get('content', '')

                            total_response_time += (end_time - start_time)
                            total_prompt_length += len(prompt)
                            total_response_length += len(content)
                            # Rough token estimation (1 token ≈ 4 characters)
                            total_tokens += len(content) // 4
                            successful_runs += 1

                    except Exception as e:
                        logger.warning(f"Performance test run failed: {e}")
                        continue

            # Restore original model
            self.config.model_name = original_model

            if successful_runs > 0:
                avg_response_time = total_response_time / successful_runs
                tokens_per_second = total_tokens / total_response_time if total_response_time > 0 else 0

                return ModelPerformanceMetrics(
                    model_name=target_model,
                    avg_response_time=avg_response_time,
                    tokens_per_second=tokens_per_second,
                    test_prompt_length=total_prompt_length // successful_runs,
                    test_response_length=total_response_length // successful_runs
                )
            else:
                return ModelPerformanceMetrics(
                    model_name=target_model,
                    avg_response_time=0.0,
                    tokens_per_second=0.0
                )

        except Exception as e:
            # Restore original model in case of error
            self.config.model_name = original_model
            logger.error(f"Performance measurement failed: {e}")
            return ModelPerformanceMetrics(
                model_name=target_model,
                avg_response_time=0.0,
                tokens_per_second=0.0
            )

    def get_health_summary(self, results: dict[str, HealthCheckResult]) -> dict[str, any]:
        """Generate a summary of health check results.

        Args:
            results: Dictionary of health check results

        Returns:
            Summary dictionary with overall status and details
        """
        all_healthy = all(result.is_healthy for result in results.values())
        total_checks = len(results)
        healthy_checks = sum(1 for result in results.values() if result.is_healthy)

        return {
            "overall_healthy": all_healthy,
            "health_score": healthy_checks / total_checks if total_checks > 0 else 0.0,
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "failed_checks": [name for name, result in results.items() if not result.is_healthy],
            "avg_response_time": sum(result.response_time for result in results.values()) / total_checks if total_checks > 0 else 0.0,
            "timestamp": time.time()
        }
