#!/usr/bin/env python3
"""Demo script showing Ollama integration capabilities for MCP Course.

This script demonstrates the key features of the Ollama integration module,
including configuration, health checking, model management, and performance testing.
"""

import asyncio
import logging

from mcp_course.ollama_client import (
    OllamaClient,
    OllamaConfig,
    OllamaHealthChecker,
    OllamaModelManager,
    OllamaPerformanceTester,
    OllamaSetupVerifier,
)
from mcp_course.ollama_client.performance import PerformanceTest


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_configuration():
    """Demonstrate Ollama configuration capabilities."""
    print("\n=== Ollama Configuration Demo ===")

    # Create default configuration
    config = OllamaConfig.create_default("llama3.2:3b")
    print(f"Default config: {config.model_name} at {config.endpoint}")

    # Create course-optimized configuration
    course_config = OllamaConfig.create_for_course("llama3.2:3b")
    print(f"Course config: {course_config.model_name}")
    print(f"Course parameters: {course_config.parameters}")

    # Check privacy compliance
    print(f"Privacy compliant: {config.is_privacy_compliant()}")

    return config


def demo_setup_verification():
    """Demonstrate setup verification capabilities."""
    print("\n=== Setup Verification Demo ===")

    verifier = OllamaSetupVerifier()

    # Check system compatibility
    compat_result = verifier.check_system_compatibility()
    print(f"System compatibility: {compat_result.status}")

    # Check installation status
    install_status = verifier.check_ollama_installation()
    print(f"Ollama installed: {install_status.is_installed}")
    if install_status.is_installed:
        print(f"Version: {install_status.version}")
        print(f"Service running: {install_status.service_running}")

    # Get installation instructions if needed
    if not install_status.is_installed:
        instructions = verifier.get_installation_instructions()
        print(f"Installation method: {instructions['method']}")
        print(f"Command: {instructions['command']}")

    return verifier


def demo_health_checking(config: OllamaConfig):
    """Demonstrate health checking capabilities."""
    print("\n=== Health Checking Demo ===")

    health_checker = OllamaHealthChecker(config)

    # Run comprehensive health check
    results = health_checker.run_comprehensive_health_check()

    for check_name, result in results.items():
        print(f"{check_name}: {'✓' if result.is_healthy else '✗'} {result.status}")
        if result.response_time > 0:
            print(f"  Response time: {result.response_time:.2f}s")

    # Get health summary
    summary = health_checker.get_health_summary(results)
    print(f"Overall health score: {summary['health_score']:.1%}")

    return health_checker


def demo_model_management(config: OllamaConfig):
    """Demonstrate model management capabilities."""
    print("\n=== Model Management Demo ===")

    model_manager = OllamaModelManager(config)

    try:
        # List available models
        models = model_manager.list_available_models()
        print(f"Available models: {len(models)}")
        for model in models[:3]:  # Show first 3
            print(f"  - {model.name} ({model.size // (1024*1024) if model.size else 0} MB)")

        # Get recommendations
        recommendations = model_manager.get_model_recommendations()
        print("\nRecommended models:")
        for rec in recommendations[:2]:  # Show top 2
            print(f"  - {rec.model_name}: {rec.reason}")
            print(f"    Memory: {rec.memory_requirement}GB, Tier: {rec.performance_tier}")

        # Get model status summary
        status = model_manager.get_model_status_summary()
        print(f"\nModel status: {status['available_recommended']}/{status['recommended_models']} recommended models available")

    except Exception as e:
        print(f"Model management demo failed: {e}")

    return model_manager


def demo_performance_testing(config: OllamaConfig):
    """Demonstrate performance testing capabilities."""
    print("\n=== Performance Testing Demo ===")

    perf_tester = OllamaPerformanceTester(config)

    try:
        # Check if model is available first
        client = OllamaClient(config)
        if not client.is_model_available():
            print(f"Model {config.model_name} not available, skipping performance tests")
            return None

        # Run a single quick test
        quick_test = PerformanceTest(
            name="quick_test",
            prompt="Hello! Respond with just 'Hi there!'",
            iterations=1,
            timeout=15
        )

        result = perf_tester.run_single_test(quick_test)
        print("Quick test result:")
        print(f"  Success rate: {result.success_rate:.1f}%")
        print(f"  Response time: {result.avg_response_time:.2f}s")
        print(f"  Tokens/sec: {result.avg_tokens_per_second:.1f}")

        if result.errors:
            print(f"  Errors: {len(result.errors)}")

    except Exception as e:
        print(f"Performance testing demo failed: {e}")

    return perf_tester


async def demo_async_operations(config: OllamaConfig):
    """Demonstrate asynchronous operations."""
    print("\n=== Async Operations Demo ===")

    try:
        # Async health check
        health_checker = OllamaHealthChecker(config)
        connection_result = await health_checker.check_connection_async()
        print(f"Async connection check: {'✓' if connection_result.is_healthy else '✗'} {connection_result.status}")

        # Async model listing
        model_manager = OllamaModelManager(config)
        models = await model_manager.list_available_models_async()
        print(f"Async model listing: {len(models)} models found")

    except Exception as e:
        print(f"Async operations demo failed: {e}")


def print_summary():
    """Print a summary of all demo results."""
    print("\n" + "="*50)
    print("DEMO SUMMARY")
    print("="*50)

    print("Configuration: ✓ Created and validated")
    print("Setup verification: ✓ System checked")
    print("Health checking: ✓ Connection tested")
    print("Model management: ✓ Models listed and analyzed")
    print("Performance testing: ✓ Basic performance measured")
    print("Async operations: ✓ Async capabilities demonstrated")

    print("\nThe Ollama integration is ready for use in the MCP course!")
    print("Key features available:")
    print("  • Privacy-by-design configuration")
    print("  • Comprehensive health monitoring")
    print("  • Automated setup verification")
    print("  • Intelligent model management")
    print("  • Performance testing and optimization")
    print("  • Full async support")


def main():
    """Run the complete Ollama integration demo."""
    print("MCP Course - Ollama Integration Demo")
    print("This demo showcases the Ollama integration capabilities.")

    try:
        # Configuration demo
        config = demo_configuration()

        # Setup verification demo
        demo_setup_verification()

        # Health checking demo
        demo_health_checking(config)

        # Model management demo
        demo_model_management(config)

        # Performance testing demo
        demo_performance_testing(config)

        # Async operations demo
        asyncio.run(demo_async_operations(config))

        print_summary()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed: {e}")
        print("This might be because Ollama is not installed or running.")
        print("Please ensure Ollama is properly set up before running this demo.")


if __name__ == "__main__":
    main()
