"""Performance testing and optimization tools for Ollama models.

This module provides comprehensive performance testing, monitoring, and
optimization utilities for Ollama models in the MCP course context.
"""

import asyncio
from dataclasses import dataclass, field
import logging
import statistics
import time
from typing import ClassVar

from mcp_course.ollama_client.client import OllamaClient
from mcp_course.ollama_client.config import OllamaConfig


logger = logging.getLogger(__name__)

HIGH_SUCCESS_RATE: int = 100
LOW_SUCCESS_RATE: int = 90
EXCELLENT_AVG_RESPONSE_TIME: int = 2
GOOD_AVG_RESPONSE_TIME: int = 5
FAIR_AVG_RESPONSE_TIME: int = 10
LOW_AVG_RESPONSE_TIME: int = 30
LOW_TOKEN_GENERATION_RATE: int = 10
QUALITY_CODE_SUCCESS_RATE: int = 80

@dataclass
class PerformanceTest:
    """Configuration for a performance test.

    Attributes:
        name: Test name
        prompt: Test prompt
        expected_response_type: Expected type of response
        timeout: Test timeout in seconds
        iterations: Number of test iterations
        parameters: Model parameters for this test
    """
    name: str
    prompt: str
    expected_response_type: str = "text"
    timeout: int = 30
    iterations: int = 3
    parameters: dict[str, any] = field(default_factory=dict)


@dataclass
class PerformanceResult:
    """Result of a performance test.

    Attributes:
        test_name: Name of the test
        model_name: Model that was tested
        success_rate: Percentage of successful iterations
        avg_response_time: Average response time in seconds
        min_response_time: Minimum response time
        max_response_time: Maximum response time
        std_response_time: Standard deviation of response times
        avg_tokens_per_second: Average tokens generated per second
        total_tokens: Total tokens generated
        errors: List of errors encountered
        response_quality_score: Quality score (0-10) if applicable
    """
    test_name: str
    model_name: str
    success_rate: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    std_response_time: float
    avg_tokens_per_second: float
    total_tokens: int
    errors: list[str] = field(default_factory=list)
    response_quality_score: float | None = None


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation based on performance analysis.

    Attributes:
        category: Category of optimization (memory, speed, quality)
        priority: Priority level (high, medium, low)
        description: Description of the recommendation
        implementation: How to implement the recommendation
        expected_improvement: Expected improvement description
    """
    category: str
    priority: str
    description: str
    implementation: str
    expected_improvement: str


class OllamaPerformanceTester:
    """Comprehensive performance testing for Ollama models.

    This class provides tools for testing model performance, analyzing results,
    and generating optimization recommendations for the MCP course platform.
    """

    # Standard test suite for course models
    STANDARD_TESTS: ClassVar[list[PerformanceTest]] = [
        PerformanceTest(
            name="basic_response",
            prompt="Hello! Please respond with a brief greeting.",
            expected_response_type="greeting",
            iterations=5
        ),
        PerformanceTest(
            name="mcp_explanation",
            prompt="Explain what MCP (Model Context Protocol) is in 2-3 sentences.",
            expected_response_type="explanation",
            timeout=45,
            iterations=3
        ),
        PerformanceTest(
            name="code_generation",
            prompt="Write a simple Python function that adds two numbers and returns the result.",
            expected_response_type="code",
            timeout=60,
            iterations=3
        ),
        PerformanceTest(
            name="json_response",
            prompt="Return a JSON object with three fields: name, age, and city. Use example values.",
            expected_response_type="json",
            parameters={"temperature": 0.1},  # Lower temperature for structured output
            iterations=3
        ),
        PerformanceTest(
            name="long_context",
            prompt="Given this context about MCP: " + "MCP is a protocol for connecting AI models with external tools and data sources. " * 10 + " Now explain the main benefits of using MCP.",
            expected_response_type="analysis",
            timeout=90,
            iterations=2
        )
    ]

    def __init__(self, config: OllamaConfig):
        """Initialize the performance tester.

        Args:
            config: Ollama configuration
        """
        self.config = config
        self.client = OllamaClient(config)

    def run_single_test(self, test: PerformanceTest, model_name: str | None = None) -> PerformanceResult:
        """Run a single performance test.

        Args:
            test: Performance test configuration
            model_name: Model to test (defaults to config model)

        Returns:
            PerformanceResult with test results
        """
        target_model = model_name or self.config.model_name
        response_times = []
        token_counts = []
        errors = []
        successful_iterations = 0

        logger.info(f"Running test '{test.name}' on model '{target_model}'")

        # Temporarily update model and parameters
        original_model = self.config.model_name
        original_params = self.config.parameters.copy()

        self.config.model_name = target_model
        if test.parameters:
            self.config.parameters.update(test.parameters)

        try:
            for iteration in range(test.iterations):
                try:
                    start_time = time.time()

                    response = self.client.generate(
                        prompt=test.prompt,
                        timeout=test.timeout
                    )

                    end_time = time.time()
                    response_time = end_time - start_time

                    if response and 'message' in response:
                        content = response['message'].get('content', '')
                        if content:
                            response_times.append(response_time)
                            # Rough token estimation (1 token ≈ 4 characters)
                            token_count = len(content) // 4
                            token_counts.append(token_count)
                            successful_iterations += 1
                        else:
                            errors.append(f"Iteration {iteration + 1}: Empty response")
                    else:
                        errors.append(f"Iteration {iteration + 1}: Invalid response format")

                except Exception as e:
                    errors.append(f"Iteration {iteration + 1}: {e!s}")

        finally:
            # Restore original configuration
            self.config.model_name = original_model
            self.config.parameters = original_params

        # Calculate statistics
        success_rate = (successful_iterations / test.iterations) * 100 if test.iterations > 0 else 0

        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        else:
            avg_response_time = min_response_time = max_response_time = std_response_time = 0

        total_tokens = sum(token_counts)
        total_time = sum(response_times)
        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        return PerformanceResult(
            test_name=test.name,
            model_name=target_model,
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            std_response_time=std_response_time,
            avg_tokens_per_second=avg_tokens_per_second,
            total_tokens=total_tokens,
            errors=errors
        )

    async def run_single_test_async(self, test: PerformanceTest, model_name: str | None = None) -> PerformanceResult:
        """Run a single performance test asynchronously.

        Args:
            test: Performance test configuration
            model_name: Model to test (defaults to config model)

        Returns:
            PerformanceResult with test results
        """
        target_model = model_name or self.config.model_name
        response_times = []
        token_counts = []
        errors = []
        successful_iterations = 0

        logger.info(f"Running async test '{test.name}' on model '{target_model}'")

        # Temporarily update model and parameters
        original_model = self.config.model_name
        original_params = self.config.parameters.copy()

        self.config.model_name = target_model
        if test.parameters:
            self.config.parameters.update(test.parameters)

        try:
            for iteration in range(test.iterations):
                try:
                    start_time = time.time()

                    response = await self.client.generate_async(
                        prompt=test.prompt
                    )

                    end_time = time.time()
                    response_time = end_time - start_time

                    if response and 'message' in response:
                        content = response['message'].get('content', '')
                        if content:
                            response_times.append(response_time)
                            # Rough token estimation (1 token ≈ 4 characters)
                            token_count = len(content) // 4
                            token_counts.append(token_count)
                            successful_iterations += 1
                        else:
                            errors.append(f"Iteration {iteration + 1}: Empty response")
                    else:
                        errors.append(f"Iteration {iteration + 1}: Invalid response format")

                except Exception as e:
                    errors.append(f"Iteration {iteration + 1}: {e!s}")

        finally:
            # Restore original configuration
            self.config.model_name = original_model
            self.config.parameters = original_params

        # Calculate statistics (same as sync version)
        success_rate = (successful_iterations / test.iterations) * 100 if test.iterations > 0 else 0

        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        else:
            avg_response_time = min_response_time = max_response_time = std_response_time = 0

        total_tokens = sum(token_counts)
        total_time = sum(response_times)
        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        return PerformanceResult(
            test_name=test.name,
            model_name=target_model,
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            std_response_time=std_response_time,
            avg_tokens_per_second=avg_tokens_per_second,
            total_tokens=total_tokens,
            errors=errors
        )

    def run_test_suite(self, tests: list[PerformanceTest] | None = None, model_name: str | None = None) -> list[PerformanceResult]:
        """Run a suite of performance tests.

        Args:
            tests: List of tests to run (defaults to standard tests)
            model_name: Model to test (defaults to config model)

        Returns:
            List of PerformanceResult objects
        """
        if tests is None:
            tests = self.STANDARD_TESTS

        results = []
        target_model = model_name or self.config.model_name

        logger.info(f"Running performance test suite on model '{target_model}' ({len(tests)} tests)")

        for test in tests:
            try:
                result = self.run_single_test(test, target_model)
                results.append(result)
                logger.info(f"Completed test '{test.name}': {result.success_rate:.1f}% success, {result.avg_response_time:.2f}s avg")
            except Exception as e:
                logger.error(f"Test '{test.name}' failed: {e}")
                # Create a failed result
                failed_result = PerformanceResult(
                    test_name=test.name,
                    model_name=target_model,
                    success_rate=0.0,
                    avg_response_time=0.0,
                    min_response_time=0.0,
                    max_response_time=0.0,
                    std_response_time=0.0,
                    avg_tokens_per_second=0.0,
                    total_tokens=0,
                    errors=[f"Test execution failed: {e}"]
                )
                results.append(failed_result)

        return results

    async def run_test_suite_async(self, tests: list[PerformanceTest] | None = None, model_name: str | None = None) -> list[PerformanceResult]:
        """Run a suite of performance tests asynchronously.

        Args:
            tests: List of tests to run (defaults to standard tests)
            model_name: Model to test (defaults to config model)

        Returns:
            List of PerformanceResult objects
        """
        if tests is None:
            tests = self.STANDARD_TESTS

        target_model = model_name or self.config.model_name

        logger.info(f"Running async performance test suite on model '{target_model}' ({len(tests)} tests)")

        # Run tests concurrently
        tasks = [self.run_single_test_async(test, target_model) for test in tests]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Test '{tests[i].name}' failed: {result}")
                    # Create a failed result
                    failed_result = PerformanceResult(
                        test_name=tests[i].name,
                        model_name=target_model,
                        success_rate=0.0,
                        avg_response_time=0.0,
                        min_response_time=0.0,
                        max_response_time=0.0,
                        std_response_time=0.0,
                        avg_tokens_per_second=0.0,
                        total_tokens=0,
                        errors=[f"Test execution failed: {result}"]
                    )
                    processed_results.append(failed_result)
                else:
                    processed_results.append(result)
                    logger.info(f"Completed test '{result.test_name}': {result.success_rate:.1f}% success, {result.avg_response_time:.2f}s avg")

            return processed_results

        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            return []

    def analyze_performance(self, results: list[PerformanceResult]) -> dict[str, any]:
        """Analyze performance test results and generate insights.

        Args:
            results: List of performance test results

        Returns:
            Dictionary with performance analysis
        """
        if not results:
            return {"error": "No results to analyze"}

        # Calculate overall statistics
        successful_tests = [r for r in results if r.success_rate > 0]
        total_tests = len(results)
        successful_count = len(successful_tests)

        overall_success_rate = (successful_count / total_tests) * 100 if total_tests > 0 else 0

        if successful_tests:
            avg_response_times = [r.avg_response_time for r in successful_tests]
            tokens_per_second_values = [r.avg_tokens_per_second for r in successful_tests if r.avg_tokens_per_second > 0]

            overall_avg_response_time = statistics.mean(avg_response_times)
            overall_avg_tokens_per_second = statistics.mean(tokens_per_second_values) if tokens_per_second_values else 0

            fastest_test = min(successful_tests, key=lambda x: x.avg_response_time)
            slowest_test = max(successful_tests, key=lambda x: x.avg_response_time)
        else:
            overall_avg_response_time = 0
            overall_avg_tokens_per_second = 0
            fastest_test = slowest_test = None

        # Identify issues
        issues = []
        for result in results:
            if result.success_rate < HIGH_SUCCESS_RATE:
                issues.extend(result.errors)
            if result.avg_response_time > LOW_AVG_RESPONSE_TIME:
                issues.append(f"Slow response time in {result.test_name}: {result.avg_response_time:.2f}s")

        # Performance classification
        if overall_avg_response_time < EXCELLENT_AVG_RESPONSE_TIME:
            performance_tier = "Excellent"
        elif overall_avg_response_time < GOOD_AVG_RESPONSE_TIME:
            performance_tier = "Good"
        elif overall_avg_response_time < FAIR_AVG_RESPONSE_TIME:
            performance_tier = "Fair"
        else:
            performance_tier = "Poor"

        return {
            "overall_success_rate": overall_success_rate,
            "successful_tests": successful_count,
            "total_tests": total_tests,
            "overall_avg_response_time": overall_avg_response_time,
            "overall_avg_tokens_per_second": overall_avg_tokens_per_second,
            "performance_tier": performance_tier,
            "fastest_test": fastest_test.test_name if fastest_test else None,
            "slowest_test": slowest_test.test_name if slowest_test else None,
            "issues": issues,
            "test_details": [
                {
                    "name": r.test_name,
                    "success_rate": r.success_rate,
                    "avg_response_time": r.avg_response_time,
                    "tokens_per_second": r.avg_tokens_per_second,
                    "has_errors": len(r.errors) > 0
                }
                for r in results
            ]
        }

    def generate_optimization_recommendations(self, analysis: dict[str, any]) -> list[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance analysis.

        Args:
            analysis: Performance analysis results

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Response time optimizations
        if analysis.get("overall_avg_response_time", 0) > FAIR_AVG_RESPONSE_TIME:
            recommendations.append(OptimizationRecommendation(
                category="speed",
                priority="high",
                description="Response times are slow (>10s average)",
                implementation="Consider using a smaller/faster model, reduce context window, or increase system resources",
                expected_improvement="50-80% faster response times"
            ))
        elif analysis.get("overall_avg_response_time", 0) > GOOD_AVG_RESPONSE_TIME:
            recommendations.append(OptimizationRecommendation(
                category="speed",
                priority="medium",
                description="Response times could be improved (>5s average)",
                implementation="Optimize model parameters (lower temperature, smaller max_tokens)",
                expected_improvement="20-40% faster response times"
            ))

        # Success rate optimizations
        if analysis.get("overall_success_rate", 0) < LOW_SUCCESS_RATE:
            recommendations.append(OptimizationRecommendation(
                category="reliability",
                priority="high",
                description="Low success rate indicates reliability issues",
                implementation="Check Ollama server stability, increase timeouts, verify model integrity",
                expected_improvement="Improved reliability and consistency"
            ))

        # Token generation optimizations
        if analysis.get("overall_avg_tokens_per_second", 0) < LOW_TOKEN_GENERATION_RATE:
            recommendations.append(OptimizationRecommendation(
                category="throughput",
                priority="medium",
                description="Low token generation rate",
                implementation="Ensure sufficient system memory, close other applications, consider GPU acceleration",
                expected_improvement="2-5x improvement in token generation speed"
            ))

        # Memory optimizations
        recommendations.append(OptimizationRecommendation(
            category="memory",
            priority="low",
            description="General memory optimization",
            implementation="Monitor system memory usage, consider model quantization, use streaming responses",
            expected_improvement="Reduced memory footprint and better stability"
        ))

        # Course-specific optimizations
        if "code_generation" in [detail["name"] for detail in analysis.get("test_details", [])]:
            code_test = next((detail for detail in analysis.get("test_details", []) if detail["name"] == "code_generation"), None)
            if code_test and code_test.get("success_rate", 0) < QUALITY_CODE_SUCCESS_RATE:
                recommendations.append(OptimizationRecommendation(
                    category="quality",
                    priority="medium",
                    description="Code generation performance needs improvement",
                    implementation="Use code-specialized models (e.g., codellama), adjust temperature for more deterministic output",
                    expected_improvement="Better code quality and consistency"
                ))

        return recommendations

    def create_performance_report(self, results: list[PerformanceResult]) -> dict[str, any]:
        """Create a comprehensive performance report.

        Args:
            results: List of performance test results

        Returns:
            Dictionary with complete performance report
        """
        analysis = self.analyze_performance(results)
        recommendations = self.generate_optimization_recommendations(analysis)

        return {
            "model_name": results[0].model_name if results else "Unknown",
            "test_timestamp": time.time(),
            "summary": analysis,
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "success_rate": r.success_rate,
                    "avg_response_time": r.avg_response_time,
                    "min_response_time": r.min_response_time,
                    "max_response_time": r.max_response_time,
                    "std_response_time": r.std_response_time,
                    "avg_tokens_per_second": r.avg_tokens_per_second,
                    "total_tokens": r.total_tokens,
                    "error_count": len(r.errors),
                    "errors": r.errors[:5]  # Limit to first 5 errors
                }
                for r in results
            ],
            "optimization_recommendations": [
                {
                    "category": rec.category,
                    "priority": rec.priority,
                    "description": rec.description,
                    "implementation": rec.implementation,
                    "expected_improvement": rec.expected_improvement
                }
                for rec in recommendations
            ]
        }
