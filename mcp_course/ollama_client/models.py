"""Ollama model management utilities for MCP Course.

This module provides comprehensive model management capabilities including
downloading, updating, performance testing, and optimization for the course platform.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
from typing import ClassVar

from mcp_course.ollama_client.client import OllamaClient, OllamaConnectionError, OllamaModelError
from mcp_course.ollama_client.config import OllamaConfig
from mcp_course.ollama_client.health import ModelPerformanceMetrics, OllamaHealthChecker


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an Ollama model.

    Attributes:
        name: Model name
        size: Model size in bytes
        digest: Model digest/hash
        modified: Last modified timestamp
        family: Model family (e.g., llama, mistral)
        parameter_count: Number of parameters (if known)
        quantization: Quantization level (if applicable)
        recommended_memory: Recommended memory in GB
        description: Model description
    """
    name: str
    size: int = 0
    digest: str = ""
    modified: str = ""
    family: str = ""
    parameter_count: str | None = None
    quantization: str | None = None
    recommended_memory: float | None = None
    description: str = ""


@dataclass
class ModelRecommendation:
    """Recommendation for model selection.

    Attributes:
        model_name: Recommended model name
        reason: Reason for recommendation
        use_case: Primary use case
        memory_requirement: Memory requirement in GB
        performance_tier: Performance tier (fast/balanced/quality)
        course_suitability: Suitability score for course (0-10)
    """
    model_name: str
    reason: str
    use_case: str
    memory_requirement: float
    performance_tier: str
    course_suitability: int


class OllamaModelManager:
    """Comprehensive model management for Ollama.

    This class provides tools for discovering, downloading, managing, and
    optimizing Ollama models for the MCP course platform.
    """

    # Course-recommended models with their characteristics
    COURSE_MODELS: ClassVar[dict[str, dict[str, any]]] = {
        "llama3.2:1b": {
            "family": "llama",
            "parameter_count": "1B",
            "recommended_memory": 2.0,
            "description": "Fast, lightweight model perfect for learning and development",
            "course_suitability": 9,
            "performance_tier": "fast"
        },
        "llama3.2:3b": {
            "family": "llama",
            "parameter_count": "3B",
            "recommended_memory": 4.0,
            "description": "Balanced model with good performance and reasonable resource usage",
            "course_suitability": 10,
            "performance_tier": "balanced"
        },
        "mistral": {
            "family": "mistral",
            "parameter_count": "7B",
            "recommended_memory": 6.0,
            "description": "Alternative model family for comparison and learning",
            "course_suitability": 7,
            "performance_tier": "balanced"
        },
        "codellama:7b": {
            "family": "codellama",
            "parameter_count": "7B",
            "recommended_memory": 6.0,
            "description": "Specialized model for code generation and understanding",
            "course_suitability": 8,
            "performance_tier": "balanced"
        }
    }

    def __init__(self, config: OllamaConfig):
        """Initialize the model manager.

        Args:
            config: Ollama configuration
        """
        self.config = config
        self.client = OllamaClient(config)
        self.health_checker = OllamaHealthChecker(config)

    def list_available_models(self) -> list[ModelInfo]:
        """List all models available on the Ollama server.

        Returns:
            List of ModelInfo objects for available models

        Raises:
            OllamaConnectionError: If connection fails
        """
        try:
            models_data = self.client.list_models()
            models = []

            for model_data in models_data:
                model_info = ModelInfo(
                    name=model_data.get('model', ''),
                    size=model_data.get('size', 0),
                    digest=model_data.get('digest', ''),
                    modified=model_data.get('modified_at', ''),
                )

                # Enhance with course model info if available
                if model_info.name in self.COURSE_MODELS:
                    course_info = self.COURSE_MODELS[model_info.name]
                    model_info.family = course_info.get('family', '')
                    model_info.parameter_count = course_info.get('parameter_count')
                    model_info.recommended_memory = course_info.get('recommended_memory')
                    model_info.description = course_info.get('description', '')

                models.append(model_info)

            return models

        except Exception as e:
            raise OllamaConnectionError(f"Failed to list models: {e}") from e

    async def list_available_models_async(self) -> list[ModelInfo]:
        """List all models available on the Ollama server asynchronously.

        Returns:
            List of ModelInfo objects for available models

        Raises:
            OllamaConnectionError: If connection fails
        """
        try:
            models_data = await self.client.list_models_async()
            models = []

            for model_data in models_data:
                model_info = ModelInfo(
                    name=model_data.get('name', ''),
                    size=model_data.get('size', 0),
                    digest=model_data.get('digest', ''),
                    modified=model_data.get('modified_at', ''),
                )

                # Enhance with course model info if available
                if model_info.name in self.COURSE_MODELS:
                    course_info = self.COURSE_MODELS[model_info.name]
                    model_info.family = course_info.get('family', '')
                    model_info.parameter_count = course_info.get('parameter_count')
                    model_info.recommended_memory = course_info.get('recommended_memory')
                    model_info.description = course_info.get('description', '')

                models.append(model_info)

            return models

        except Exception as e:
            raise OllamaConnectionError(f"Failed to list models: {e}") from e

    def get_model_recommendations(self, available_memory: float | None = None) -> list[ModelRecommendation]:
        """Get model recommendations based on system capabilities.

        Args:
            available_memory: Available system memory in GB

        Returns:
            List of ModelRecommendation objects sorted by suitability
        """
        recommendations = []

        for model_name, info in self.COURSE_MODELS.items():
            # Skip if memory requirement exceeds available memory
            if available_memory and info['recommended_memory'] > available_memory:
                continue

            recommendation = ModelRecommendation(
                model_name=model_name,
                reason=self._get_recommendation_reason(info, available_memory),
                use_case=self._get_use_case(info),
                memory_requirement=info['recommended_memory'],
                performance_tier=info['performance_tier'],
                course_suitability=info['course_suitability']
            )
            recommendations.append(recommendation)

        # Sort by course suitability (descending) and memory requirement (ascending)
        recommendations.sort(key=lambda x: (-x.course_suitability, x.memory_requirement))

        return recommendations

    def _get_recommendation_reason(self, model_info: dict, available_memory: float | None) -> str:
        """Generate recommendation reason based on model info and system capabilities."""
        good_suitability_threshold: int = 7
        excellent_suitability_threshold: int = 9

        reasons = []

        if model_info['performance_tier'] == 'fast':
            reasons.append("Fast response times")
        elif model_info['performance_tier'] == 'balanced':
            reasons.append("Good balance of speed and quality")
        elif model_info['performance_tier'] == 'quality':
            reasons.append("High-quality responses")

        if available_memory:
            if model_info['recommended_memory'] <= available_memory * 0.5:
                reasons.append("low memory usage")
            elif model_info['recommended_memory'] <= available_memory * 0.8:
                reasons.append("moderate memory usage")

        if model_info['course_suitability'] >= excellent_suitability_threshold:
            reasons.append("excellent for learning MCP")
        elif model_info['course_suitability'] >= good_suitability_threshold:
            reasons.append("good for MCP course")

        return ", ".join(reasons)

    def _get_use_case(self, model_info: dict) -> str:
        """Get primary use case for a model."""
        if model_info['family'] == 'codellama':
            return "Code generation and understanding"
        elif model_info['performance_tier'] == 'fast':
            return "Development and testing"
        elif model_info['performance_tier'] == 'balanced':
            return "General learning and course work"
        elif model_info['performance_tier'] == 'quality':
            return "Advanced projects and production use"
        else:
            return "General purpose"

    def download_model(self, model_name: str, progress_callback: Callable | None = None) -> bool:
        """Download a model to the Ollama server.

        Args:
            model_name: Name of the model to download
            progress_callback: Optional callback for progress updates

        Returns:
            True if download was successful

        Raises:
            OllamaModelError: If download fails
        """
        try:
            logger.info(f"Starting download of model: {model_name}")

            if progress_callback:
                progress_callback(f"Starting download of {model_name}...")

            success = self.client.pull_model(model_name)

            if success:
                logger.info(f"Successfully downloaded model: {model_name}")
                if progress_callback:
                    progress_callback(f"Successfully downloaded {model_name}")
                return True
            else:
                raise OllamaModelError(f"Failed to download model: {model_name}")

        except Exception as e:
            error_msg = f"Model download failed for {model_name}: {e}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(f"Error: {error_msg}")
            raise OllamaModelError(error_msg) from e

    async def download_model_async(self, model_name: str, progress_callback: Callable | None = None) -> bool:
        """Download a model to the Ollama server asynchronously.

        Args:
            model_name: Name of the model to download
            progress_callback: Optional callback for progress updates

        Returns:
            True if download was successful

        Raises:
            OllamaModelError: If download fails
        """
        try:
            logger.info(f"Starting download of model: {model_name}")

            if progress_callback:
                progress_callback(f"Starting download of {model_name}...")

            success = await self.client.pull_model_async(model_name)

            if success:
                logger.info(f"Successfully downloaded model: {model_name}")
                if progress_callback:
                    progress_callback(f"Successfully downloaded {model_name}")
                return True
            else:
                raise OllamaModelError(f"Failed to download model: {model_name}")

        except Exception as e:
            error_msg = f"Model download failed for {model_name}: {e}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(f"Error: {error_msg}")
            raise OllamaModelError(error_msg) from e

    def setup_course_models(self, memory_limit: float | None = None) -> dict[str, bool]:
        """Set up recommended models for the course.

        Args:
            memory_limit: Memory limit in GB (optional)

        Returns:
            Dictionary mapping model names to success status
        """
        results = {}
        recommendations = self.get_model_recommendations(memory_limit)

        # Download top 2 recommendations
        for recommendation in recommendations[:2]:
            try:
                logger.info(f"Setting up course model: {recommendation.model_name}")
                success = self.download_model(recommendation.model_name)
                results[recommendation.model_name] = success
            except Exception as e:
                logger.error(f"Failed to setup model {recommendation.model_name}: {e}")
                results[recommendation.model_name] = False

        return results

    async def setup_course_models_async(self, memory_limit: float | None = None) -> dict[str, bool]:
        """Set up recommended models for the course asynchronously.

        Args:
            memory_limit: Memory limit in GB (optional)

        Returns:
            Dictionary mapping model names to success status
        """
        results = {}
        recommendations = self.get_model_recommendations(memory_limit)

        # Download top 2 recommendations concurrently
        tasks = []
        model_names = []

        for recommendation in recommendations[:2]:
            model_names.append(recommendation.model_name)
            tasks.append(self.download_model_async(recommendation.model_name))

        try:
            download_results = await asyncio.gather(*tasks, return_exceptions=True)

            for model_name, result in zip(model_names, download_results, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"Failed to setup model {model_name}: {result}")
                    results[model_name] = False
                else:
                    results[model_name] = result

        except Exception as e:
            logger.error(f"Failed to setup course models: {e}")
            for model_name in model_names:
                results[model_name] = False

        return results

    def benchmark_models(self, model_names: list[str] | None = None) -> dict[str, ModelPerformanceMetrics]:
        """Benchmark performance of specified models.

        Args:
            model_names: List of model names to benchmark (defaults to available models)

        Returns:
            Dictionary mapping model names to performance metrics
        """
        if model_names is None:
            # Get available models
            try:
                available_models = self.list_available_models()
                model_names = [model.name for model in available_models]
            except Exception as e:
                logger.error(f"Failed to list models for benchmarking: {e}")
                return {}

        results = {}

        for model_name in model_names:
            try:
                logger.info(f"Benchmarking model: {model_name}")
                metrics = self.health_checker.measure_model_performance(model_name=model_name)
                results[model_name] = metrics
            except Exception as e:
                logger.error(f"Failed to benchmark model {model_name}: {e}")
                # Create empty metrics for failed benchmark
                results[model_name] = ModelPerformanceMetrics(
                    model_name=model_name,
                    avg_response_time=0.0,
                    tokens_per_second=0.0
                )

        return results

    def get_optimization_suggestions(self, model_name: str) -> list[str]:
        """Get optimization suggestions for a specific model.

        Args:
            model_name: Name of the model to optimize

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Get model info
        model_info = None
        if model_name in self.COURSE_MODELS:
            model_info = self.COURSE_MODELS[model_name]

        # General suggestions
        suggestions.extend([
            "Ensure Ollama server has sufficient memory allocated",
            "Close other memory-intensive applications while using the model",
            "Use smaller context windows for faster responses"
        ])

        # Model-specific suggestions
        if model_info:
            if model_info['performance_tier'] == 'quality':
                suggestions.extend([
                    "Consider using a smaller model variant for development",
                    "Increase system memory for better performance with large models"
                ])
            elif model_info['performance_tier'] == 'fast':
                suggestions.extend([
                    "This model is already optimized for speed",
                    "Consider using larger models for production use cases"
                ])

        # Parameter optimization suggestions
        suggestions.extend([
            "Adjust temperature parameter for more consistent responses (lower values)",
            "Reduce max_tokens for faster responses in interactive scenarios",
            "Use streaming responses for better perceived performance"
        ])

        return suggestions

    def cleanup_unused_models(self, keep_models: set[str] | None = None) -> list[str]:
        """Clean up unused models to free space.

        Args:
            keep_models: Set of model names to keep (optional)

        Returns:
            List of models that were removed
        """
        if keep_models is None:
            # Keep course-recommended models by default
            keep_models = set(self.COURSE_MODELS.keys())

        removed_models = []

        try:
            available_models = self.list_available_models()

            for model in available_models:
                if model.name not in keep_models:
                    try:
                        # Note: Ollama doesn't have a direct delete API in the Python client
                        # This would need to be implemented using subprocess calls to ollama CLI
                        logger.info(f"Would remove model: {model.name} (not implemented)")
                        logger.info(f"run `ollama rn {model.name}")
                    except Exception as e:
                        logger.error(f"Failed to remove model {model.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to cleanup models: {e}")

        return removed_models

    def get_model_status_summary(self) -> dict[str, any]:
        """Get a comprehensive summary of model status.

        Returns:
            Dictionary with model status information
        """
        try:
            available_models = self.list_available_models()
            recommendations = self.get_model_recommendations()

            # Check which recommended models are available
            available_names = {model.name for model in available_models}
            recommended_names = {rec.model_name for rec in recommendations}

            missing_recommended = recommended_names - available_names
            extra_models = available_names - recommended_names

            return {
                "total_models": len(available_models),
                "recommended_models": len(recommendations),
                "available_recommended": len(recommended_names & available_names),
                "missing_recommended": list(missing_recommended),
                "extra_models": list(extra_models),
                "models": [
                    {
                        "name": model.name,
                        "size_mb": model.size // (1024 * 1024) if model.size else 0,
                        "family": model.family,
                        "recommended": model.name in recommended_names
                    }
                    for model in available_models
                ],
                "recommendations": [
                    {
                        "model": rec.model_name,
                        "reason": rec.reason,
                        "memory_gb": rec.memory_requirement,
                        "available": rec.model_name in available_names
                    }
                    for rec in recommendations
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get model status summary: {e}")
            return {
                "error": str(e),
                "total_models": 0,
                "recommended_models": 0,
                "available_recommended": 0
            }
