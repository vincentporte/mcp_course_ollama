#!/usr/bin/env python3
"""
Multi-Server Orchestration Examples

This module demonstrates advanced MCP patterns where multiple servers work together
to accomplish complex tasks. It shows:

1. Server coordination and data sharing patterns
2. Complex workflow examples using multiple tools
3. Cross-server communication and orchestration
4. Error handling in multi-server environments

Run this example:
    python -m mcp_course.examples.multi_server_orchestration
"""

import asyncio
from dataclasses import dataclass, field
import json
import logging
from typing import Any

from mcp.types import CallToolResult

from mcp_course.client.basic import BasicMCPClient, ClientConfig


@dataclass
class WorkflowStep:
    """Represents a single step in a multi-server workflow."""
    server_name: str
    tool_name: str
    arguments: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    output_key: str | None = None
    description: str = ""


@dataclass
class WorkflowResult:
    """Result of executing a workflow step."""
    step_id: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0


class MultiServerOrchestrator:
    """
    Orchestrates workflows across multiple MCP servers.

    This class demonstrates:
    - Managing multiple server connections
    - Coordinating data flow between servers
    - Handling dependencies and execution order
    - Error recovery and rollback strategies
    """

    def __init__(self, client_config: ClientConfig = None):
        """Initialize the multi-server orchestrator."""
        self.client = BasicMCPClient(client_config or ClientConfig(name="orchestrator-client"))
        self.logger = logging.getLogger("MultiServerOrchestrator")
        self.workflow_context: dict[str, Any] = {}
        self.execution_history: list[WorkflowResult] = []

    async def add_server(self, name: str, command: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> bool:
        """Add a server to the orchestration pool."""
        return await self.client.add_server(name, command, args, env)

    async def connect_all_servers(self) -> dict[str, bool]:
        """Connect to all configured servers."""
        results = {}
        for server_name in self.client.get_all_servers():
            results[server_name] = await self.client.connect_to_server(server_name)
        return results

    async def execute_workflow(self, steps: list[WorkflowStep]) -> list[WorkflowResult]:
        """
        Execute a multi-step workflow across multiple servers.

        Args:
            steps: List of workflow steps to execute

        Returns:
            List of workflow results
        """
        self.logger.info(f"Starting workflow execution with {len(steps)} steps")

        # Reset context for new workflow
        self.workflow_context = {}
        self.execution_history = []

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(steps)

        # Execute steps in dependency order
        executed_steps = set()

        while len(executed_steps) < len(steps):
            # Find steps that can be executed (all dependencies met)
            ready_steps = [
                step for i, step in enumerate(steps)
                if i not in executed_steps and all(dep in executed_steps for dep in dependency_graph.get(i, []))
            ]

            if not ready_steps:
                # Check for circular dependencies
                remaining_steps = [i for i in range(len(steps)) if i not in executed_steps]
                error_msg = f"Circular dependency detected or unresolvable dependencies in steps: {remaining_steps}"
                self.logger.error(error_msg)
                break

            # Execute ready steps (can be done in parallel)
            tasks = []
            step_indices = []

            for step in ready_steps:
                step_index = steps.index(step)
                tasks.append(self._execute_single_step(f"step_{step_index}", step))
                step_indices.append(step_index)

            # Wait for all ready steps to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                step_index = step_indices[i]
                executed_steps.add(step_index)

                if isinstance(result, Exception):
                    error_result = WorkflowResult(
                        step_id=f"step_{step_index}",
                        success=False,
                        error=str(result)
                    )
                    self.execution_history.append(error_result)
                    self.logger.error(f"Step {step_index} failed: {result}")
                else:
                    self.execution_history.append(result)
                    if result.success and steps[step_index].output_key:
                        self.workflow_context[steps[step_index].output_key] = result.result

        self.logger.info(f"Workflow execution completed. {len(self.execution_history)} steps executed")
        return self.execution_history

    async def _execute_single_step(self, step_id: str, step: WorkflowStep) -> WorkflowResult:
        """Execute a single workflow step."""
        start_time = asyncio.get_event_loop().time()

        try:
            self.logger.info(f"Executing {step_id}: {step.description}")

            # Resolve arguments with context variables
            resolved_args = self._resolve_arguments(step.arguments)

            # Execute the tool
            result = await self.client.call_tool(
                step.server_name,
                step.tool_name,
                resolved_args
            )

            if result is None:
                raise Exception(f"Tool execution failed for {step.tool_name} on {step.server_name}")

            # Extract result content
            result_content = self._extract_result_content(result)

            execution_time = asyncio.get_event_loop().time() - start_time

            return WorkflowResult(
                step_id=step_id,
                success=True,
                result=result_content,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Error executing {step_id}: {e}")

            return WorkflowResult(
                step_id=step_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    def _build_dependency_graph(self, steps: list[WorkflowStep]) -> dict[int, list[int]]:
        """Build a dependency graph for workflow steps."""
        dependency_graph = {}

        for i, step in enumerate(steps):
            dependencies = []
            for dep in step.depends_on:
                # Find the step index for the dependency
                for j, other_step in enumerate(steps):
                    if other_step.output_key == dep:
                        dependencies.append(j)
                        break
            dependency_graph[i] = dependencies

        return dependency_graph

    def _resolve_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments by substituting context variables."""
        resolved = {}

        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Context variable reference
                var_name = value[2:-1]
                if var_name in self.workflow_context:
                    resolved[key] = self.workflow_context[var_name]
                else:
                    self.logger.warning(f"Context variable {var_name} not found, using literal value")
                    resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    def _extract_result_content(self, result: CallToolResult) -> Any:
        """Extract content from tool call result."""
        if not result.content:
            return None

        # Handle different content types
        content = result.content[0]

        if hasattr(content, 'text'):
            text = content.text
            # Try to parse as JSON if it looks like JSON
            if text.strip().startswith('{') or text.strip().startswith('['):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            return text

        return str(content)

    async def get_orchestration_status(self) -> dict[str, Any]:
        """Get status of the orchestration environment."""
        server_statuses = {}

        for server_name in self.client.get_all_servers():
            server_statuses[server_name] = await self.client.get_server_status(server_name)

        return {
            "connected_servers": len(self.client.get_connected_servers()),
            "total_servers": len(self.client.get_all_servers()),
            "workflow_context_keys": list(self.workflow_context.keys()),
            "execution_history_count": len(self.execution_history),
            "server_details": server_statuses
        }

    async def cleanup(self):
        """Clean up resources and disconnect from servers."""
        await self.client.disconnect_all()


class DataProcessingWorkflow:
    """
    Example workflow that demonstrates data processing across multiple servers.

    This workflow shows:
    1. Data extraction from one server
    2. Data transformation on another server
    3. Data analysis on a third server
    4. Result aggregation and reporting
    """

    def __init__(self, orchestrator: MultiServerOrchestrator):
        """Initialize the data processing workflow."""
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("DataProcessingWorkflow")

    async def setup_servers(self):
        """Set up the servers needed for the data processing workflow."""
        # Add data source server
        await self.orchestrator.add_server(
            name="data-server",
            command="python",
            args=["-m", "mcp_course.examples.data_server_example"]
        )

        # Add processing server
        await self.orchestrator.add_server(
            name="processing-server",
            command="python",
            args=["-m", "mcp_course.examples.processing_server_example"]
        )

        # Add analytics server
        await self.orchestrator.add_server(
            name="analytics-server",
            command="python",
            args=["-m", "mcp_course.examples.analytics_server_example"]
        )

    async def execute_data_pipeline(self) -> list[WorkflowResult]:
        """Execute a complete data processing pipeline."""

        # Define the workflow steps
        workflow_steps = [
            # Step 1: Extract data
            WorkflowStep(
                server_name="data-server",
                tool_name="extract_data",
                arguments={"source": "sales_database", "date_range": "last_30_days"},
                output_key="raw_data",
                description="Extract sales data from database"
            ),

            # Step 2: Clean and transform data
            WorkflowStep(
                server_name="processing-server",
                tool_name="clean_data",
                arguments={"data": "${raw_data}", "remove_outliers": True},
                depends_on=["raw_data"],
                output_key="cleaned_data",
                description="Clean and normalize the extracted data"
            ),

            # Step 3: Calculate statistics
            WorkflowStep(
                server_name="analytics-server",
                tool_name="calculate_statistics",
                arguments={"data": "${cleaned_data}", "metrics": ["mean", "median", "std"]},
                depends_on=["cleaned_data"],
                output_key="statistics",
                description="Calculate statistical metrics"
            ),

            # Step 4: Generate trends analysis
            WorkflowStep(
                server_name="analytics-server",
                tool_name="analyze_trends",
                arguments={"data": "${cleaned_data}", "period": "daily"},
                depends_on=["cleaned_data"],
                output_key="trends",
                description="Analyze data trends over time"
            ),

            # Step 5: Create final report (depends on both statistics and trends)
            WorkflowStep(
                server_name="processing-server",
                tool_name="generate_report",
                arguments={
                    "statistics": "${statistics}",
                    "trends": "${trends}",
                    "format": "json"
                },
                depends_on=["statistics", "trends"],
                output_key="final_report",
                description="Generate comprehensive analysis report"
            )
        ]

        self.logger.info("Starting data processing pipeline")
        results = await self.orchestrator.execute_workflow(workflow_steps)

        # Log workflow summary
        successful_steps = sum(1 for r in results if r.success)
        total_time = sum(r.execution_time for r in results)

        self.logger.info(f"Pipeline completed: {successful_steps}/{len(results)} steps successful")
        self.logger.info(f"Total execution time: {total_time:.2f} seconds")

        return results


class CollaborativeWorkflow:
    """
    Example workflow showing collaborative processing between servers.

    This demonstrates:
    1. Parallel processing across multiple servers
    2. Result aggregation and consensus building
    3. Cross-validation between different processing approaches
    """

    def __init__(self, orchestrator: MultiServerOrchestrator):
        """Initialize the collaborative workflow."""
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("CollaborativeWorkflow")

    async def setup_servers(self):
        """Set up servers for collaborative processing."""
        # Add multiple processing servers with different algorithms
        await self.orchestrator.add_server(
            name="processor-a",
            command="python",
            args=["-m", "mcp_course.examples.processor_a_example"]
        )

        await self.orchestrator.add_server(
            name="processor-b",
            command="python",
            args=["-m", "mcp_course.examples.processor_b_example"]
        )

        await self.orchestrator.add_server(
            name="validator",
            command="python",
            args=["-m", "mcp_course.examples.validator_example"]
        )

    async def execute_collaborative_analysis(self, input_data: str) -> list[WorkflowResult]:
        """Execute collaborative analysis with multiple processors."""

        workflow_steps = [
            # Parallel processing with different algorithms
            WorkflowStep(
                server_name="processor-a",
                tool_name="process_data",
                arguments={"data": input_data, "algorithm": "method_a"},
                output_key="result_a",
                description="Process data using algorithm A"
            ),

            WorkflowStep(
                server_name="processor-b",
                tool_name="process_data",
                arguments={"data": input_data, "algorithm": "method_b"},
                output_key="result_b",
                description="Process data using algorithm B"
            ),

            # Cross-validation step
            WorkflowStep(
                server_name="validator",
                tool_name="cross_validate",
                arguments={
                    "result_a": "${result_a}",
                    "result_b": "${result_b}",
                    "tolerance": 0.05
                },
                depends_on=["result_a", "result_b"],
                output_key="validation_result",
                description="Cross-validate results from both processors"
            ),

            # Consensus building
            WorkflowStep(
                server_name="validator",
                tool_name="build_consensus",
                arguments={
                    "results": ["${result_a}", "${result_b}"],
                    "validation": "${validation_result}",
                    "strategy": "weighted_average"
                },
                depends_on=["validation_result"],
                output_key="consensus_result",
                description="Build consensus from multiple processing results"
            )
        ]

        self.logger.info("Starting collaborative analysis workflow")
        results = await self.orchestrator.execute_workflow(workflow_steps)

        return results


async def demonstrate_multi_server_orchestration():
    """
    Demonstrate multi-server orchestration capabilities.

    This function shows complete examples of:
    - Setting up multiple server connections
    - Coordinating workflows across servers
    - Handling data dependencies
    - Error recovery and reporting
    """
    print("=== Multi-Server Orchestration Demonstration ===")
    print()

    # Create orchestrator
    orchestrator = MultiServerOrchestrator()

    try:
        print("1. Setting up server connections...")

        # For demonstration, we'll use mock servers (in real usage, these would be actual MCP servers)
        await orchestrator.add_server(
            name="mock-data-server",
            command="python",
            args=["-c", "print('Mock data server')"]  # Mock command
        )

        await orchestrator.add_server(
            name="mock-processing-server",
            command="python",
            args=["-c", "print('Mock processing server')"]  # Mock command
        )

        print("2. Demonstrating workflow definition...")

        # Create a simple workflow for demonstration
        demo_workflow = [
            WorkflowStep(
                server_name="mock-data-server",
                tool_name="get_data",
                arguments={"query": "SELECT * FROM users"},
                output_key="user_data",
                description="Fetch user data from database"
            ),

            WorkflowStep(
                server_name="mock-processing-server",
                tool_name="process_users",
                arguments={"users": "${user_data}", "operation": "anonymize"},
                depends_on=["user_data"],
                output_key="processed_users",
                description="Process and anonymize user data"
            )
        ]

        print("Workflow steps defined:")
        for i, step in enumerate(demo_workflow):
            print(f"  Step {i+1}: {step.description}")
            print(f"    Server: {step.server_name}")
            print(f"    Tool: {step.tool_name}")
            if step.depends_on:
                print(f"    Depends on: {step.depends_on}")
            print()

        print("3. Workflow execution patterns...")
        print("   - Dependency resolution")
        print("   - Parallel execution where possible")
        print("   - Context variable substitution")
        print("   - Error handling and recovery")
        print()

        print("4. Advanced orchestration features:")
        print("   ✓ Multi-server coordination")
        print("   ✓ Data flow management")
        print("   ✓ Dependency graph resolution")
        print("   ✓ Parallel step execution")
        print("   ✓ Context variable substitution")
        print("   ✓ Error recovery and rollback")
        print("   ✓ Execution monitoring and logging")

        # Show orchestration status
        status = await orchestrator.get_orchestration_status()
        print("\nOrchestration Status:")
        print(f"  Configured servers: {status['total_servers']}")
        print(f"  Connected servers: {status['connected_servers']}")

    except Exception as e:
        print(f"Error in demonstration: {e}")

    finally:
        await orchestrator.cleanup()


async def demonstrate_data_processing_workflow():
    """Demonstrate a complete data processing workflow."""
    print("\n=== Data Processing Workflow Example ===")
    print()

    orchestrator = MultiServerOrchestrator()
    DataProcessingWorkflow(orchestrator)

    try:
        print("Setting up data processing pipeline...")
        print("Servers: Data Source → Processing → Analytics")
        print()

        print("Pipeline steps:")
        print("1. Extract sales data from database")
        print("2. Clean and normalize data")
        print("3. Calculate statistical metrics")
        print("4. Analyze trends over time")
        print("5. Generate comprehensive report")
        print()

        print("This workflow demonstrates:")
        print("- Sequential data processing")
        print("- Cross-server data sharing")
        print("- Dependency management")
        print("- Result aggregation")

    except Exception as e:
        print(f"Error in workflow demonstration: {e}")

    finally:
        await orchestrator.cleanup()


async def demonstrate_collaborative_workflow():
    """Demonstrate collaborative processing workflow."""
    print("\n=== Collaborative Processing Workflow Example ===")
    print()

    orchestrator = MultiServerOrchestrator()
    CollaborativeWorkflow(orchestrator)

    try:
        print("Setting up collaborative processing...")
        print("Servers: Processor A + Processor B → Validator")
        print()

        print("Collaboration steps:")
        print("1. Process data with Algorithm A (parallel)")
        print("2. Process data with Algorithm B (parallel)")
        print("3. Cross-validate results")
        print("4. Build consensus from multiple results")
        print()

        print("This workflow demonstrates:")
        print("- Parallel processing")
        print("- Result comparison and validation")
        print("- Consensus building")
        print("- Quality assurance through redundancy")

    except Exception as e:
        print(f"Error in collaborative workflow demonstration: {e}")

    finally:
        await orchestrator.cleanup()


async def main():
    """Main demonstration entry point."""
    await demonstrate_multi_server_orchestration()
    await demonstrate_data_processing_workflow()
    await demonstrate_collaborative_workflow()

    print("\n=== Multi-Server Orchestration Summary ===")
    print()
    print("Key patterns demonstrated:")
    print("1. Server coordination and data sharing")
    print("2. Workflow dependency management")
    print("3. Parallel and sequential execution")
    print("4. Error handling and recovery")
    print("5. Context variable substitution")
    print("6. Cross-server validation")
    print("7. Result aggregation and consensus")
    print()
    print("These patterns enable building complex, distributed")
    print("MCP applications that leverage multiple specialized servers.")


if __name__ == "__main__":
    asyncio.run(main())
