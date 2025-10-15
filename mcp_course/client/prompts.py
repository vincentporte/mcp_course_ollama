"""Prompt engineering utilities for MCP tool usage with Ollama."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from typing import Any

from mcp_course.client.integration import MCPToolDefinition


class PromptStrategy(Enum):
    """Different strategies for prompt engineering with MCP tools."""
    DIRECT = "direct"  # Direct tool usage instructions
    CONVERSATIONAL = "conversational"  # Natural conversation with tool awareness
    STEP_BY_STEP = "step_by_step"  # Guided step-by-step tool usage
    PROBLEM_SOLVING = "problem_solving"  # Problem-solving oriented prompts


@dataclass
class PromptTemplate:
    """Template for generating prompts with MCP tool integration."""
    name: str
    description: str
    template: str
    strategy: PromptStrategy
    variables: list[str] = field(default_factory=list)
    tool_integration: bool = True
    examples: list[dict[str, str]] = field(default_factory=list)


class PromptEngineering:
    """
    Prompt engineering utilities for MCP tool usage.

    This class provides sophisticated prompt engineering capabilities
    to help LLMs effectively use MCP tools, including:
    - Template-based prompt generation
    - Tool-aware prompt enhancement
    - Context-sensitive prompt adaptation
    - Multi-turn conversation prompting
    """

    def __init__(self):
        """Initialize prompt engineering utilities."""
        self.logger = logging.getLogger("PromptEngineering")
        self.templates: dict[str, PromptTemplate] = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self) -> None:
        """Initialize default prompt templates."""

        # Direct tool usage template
        self.templates["direct_tool"] = PromptTemplate(
            name="direct_tool",
            description="Direct instructions for tool usage",
            template="""You have access to the following tools:

{tool_list}

User request: {user_message}

Please use the appropriate tools to fulfill this request. Call tools when needed and provide a comprehensive response based on the results.""",
            strategy=PromptStrategy.DIRECT,
            variables=["tool_list", "user_message"]
        )

        # Conversational template
        self.templates["conversational"] = PromptTemplate(
            name="conversational",
            description="Natural conversation with tool awareness",
            template="""You are a helpful assistant with access to various tools that can help you provide better responses.

Available tools:
{tool_list}

Feel free to use these tools naturally in our conversation when they would be helpful.

User: {user_message}Ass
istant: I'll help you with that. Let me use the available tools if needed to provide you with the best possible response.""",
            strategy=PromptStrategy.CONVERSATIONAL,
            variables=["tool_list", "user_message"]
        )

        # Step-by-step template
        self.templates["step_by_step"] = PromptTemplate(
            name="step_by_step",
            description="Guided step-by-step approach",
            template="""You are helping a user accomplish a task. Break down the problem and use tools systematically.

Available tools:
{tool_list}

User request: {user_message}

Please approach this step-by-step:
1. Analyze what the user is asking for
2. Identify which tools might be helpful
3. Use the tools in a logical sequence
4. Provide a clear summary of what was accomplished""",
            strategy=PromptStrategy.STEP_BY_STEP,
            variables=["tool_list", "user_message"]
        )

        # Problem-solving template
        self.templates["problem_solving"] = PromptTemplate(
            name="problem_solving",
            description="Problem-solving oriented approach",
            template="""You are a problem-solving assistant. Analyze the user's request and use available tools to find solutions.

Available tools:
{tool_list}

Problem to solve: {user_message}

Approach:
- First, understand the problem clearly
- Consider what information or actions are needed
- Use appropriate tools to gather information or perform actions
- Synthesize the results into a solution
- Explain your reasoning and the steps taken""",
            strategy=PromptStrategy.PROBLEM_SOLVING,
            variables=["tool_list", "user_message"]
        )

    def add_template(self, template: PromptTemplate) -> None:
        """
        Add a custom prompt template.

        Args:
            template: PromptTemplate to add
        """
        self.templates[template.name] = template
        self.logger.info(f"Added prompt template: {template.name}")

    def generate_tool_list(
        self,
        tools: list[MCPToolDefinition],
        format_style: str = "detailed"
    ) -> str:
        """
        Generate a formatted list of available tools.

        Args:
            tools: List of MCP tool definitions
            format_style: Style of formatting ("simple", "detailed", "json")

        Returns:
            Formatted tool list string
        """
        if not tools:
            return "No tools available."

        if format_style == "simple":
            return "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

        elif format_style == "json":
            tool_defs = [tool.to_ollama_function() for tool in tools]
            return json.dumps(tool_defs, indent=2)

        else:  # detailed
            tool_descriptions = []
            for tool in tools:
                desc = f"**{tool.name}** (from {tool.server_name})\n"
                desc += f"  Description: {tool.description}\n"

                if tool.parameters.get("properties"):
                    desc += "  Parameters:\n"
                    for param_name, param_info in tool.parameters["properties"].items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "No description")
                        required = param_name in tool.parameters.get("required", [])
                        req_marker = " (required)" if required else " (optional)"
                        desc += f"    - {param_name} ({param_type}){req_marker}: {param_desc}\n"

                tool_descriptions.append(desc)

            return "\n".join(tool_descriptions)

    def generate_prompt(
        self,
        template_name: str,
        user_message: str,
        tools: list[MCPToolDefinition] | None = None,
        additional_context: dict[str, Any] | None = None,
        **kwargs
    ) -> str:
        """
        Generate a prompt using a template.

        Args:
            template_name: Name of the template to use
            user_message: User's message/request
            tools: Available MCP tools
            additional_context: Additional context variables
            **kwargs: Additional template variables

        Returns:
            Generated prompt string
        """
        if template_name not in self.templates:
            self.logger.error(f"Template not found: {template_name}")
            return user_message

        template = self.templates[template_name]

        # Prepare variables
        variables = {
            "user_message": user_message,
            "tool_list": self.generate_tool_list(tools or []),
            "timestamp": datetime.now().isoformat(),
        }

        # Add additional context
        if additional_context:
            variables.update(additional_context)

        # Add kwargs
        variables.update(kwargs)

        try:
            # Generate prompt from template
            prompt = template.template.format(**variables)

            self.logger.debug(f"Generated prompt using template: {template_name}")
            return prompt

        except KeyError as e:
            self.logger.error(f"Missing variable in template {template_name}: {e}")
            return user_message
        except Exception as e:
            self.logger.error(f"Error generating prompt: {e}")
            return user_message

    def enhance_prompt_with_context(
        self,
        base_prompt: str,
        conversation_history: list[dict[str, Any]] | None = None,
        tool_results: dict[str, Any] | None = None,
        user_preferences: dict[str, Any] | None = None
    ) -> str:
        """
        Enhance a prompt with additional context.

        Args:
            base_prompt: Base prompt to enhance
            conversation_history: Previous conversation messages
            tool_results: Results from previous tool calls
            user_preferences: User preferences and settings

        Returns:
            Enhanced prompt with context
        """
        enhanced_sections = [base_prompt]

        # Add conversation context
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 messages
            if recent_history:
                history_text = "Recent conversation context:\n"
                for msg in recent_history:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")[:200]  # Truncate long messages
                    history_text += f"{role}: {content}\n"
                enhanced_sections.insert(0, history_text)

        # Add tool results context
        if tool_results:
            results_text = "Previous tool results:\n"
            for tool_name, result in tool_results.items():
                result_preview = str(result)[:150]  # Truncate long results
                results_text += f"- {tool_name}: {result_preview}\n"
            enhanced_sections.insert(-1, results_text)

        # Add user preferences
        if user_preferences:
            prefs_text = "User preferences:\n"
            for pref_key, pref_value in user_preferences.items():
                prefs_text += f"- {pref_key}: {pref_value}\n"
            enhanced_sections.insert(-1, prefs_text)

        return "\n\n".join(enhanced_sections)

    def create_tool_usage_examples(
        self,
        tools: list[MCPToolDefinition],
        num_examples: int = 2
    ) -> list[dict[str, str]]:
        """
        Create example tool usage scenarios.

        Args:
            tools: Available tools
            num_examples: Number of examples to generate per tool

        Returns:
            List of example scenarios
        """
        examples = []

        for tool in tools[:3]:  # Limit to first 3 tools
            for _ in range(min(num_examples, 2)):
                # Generate example based on tool parameters
                example_args = {}
                if tool.parameters.get("properties"):
                    for param_name, param_info in tool.parameters["properties"].items():
                        param_type = param_info.get("type", "string")

                        if param_type == "string":
                            example_args[param_name] = f"example_{param_name}"
                        elif param_type == "number":
                            example_args[param_name] = 42
                        elif param_type == "boolean":
                            example_args[param_name] = True
                        elif param_type == "array":
                            example_args[param_name] = ["item1", "item2"]

                example = {
                    "user_request": f"Please use {tool.name} to {tool.description.lower()}",
                    "tool_call": f"{tool.name}({json.dumps(example_args)})",
                    "expected_outcome": f"The {tool.name} tool will execute and provide results"
                }

                examples.append(example)

        return examples

    def optimize_prompt_for_model(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int | None = None
    ) -> str:
        """
        Optimize prompt for specific model characteristics.

        Args:
            prompt: Original prompt
            model_name: Target model name
            max_tokens: Maximum token limit

        Returns:
            Optimized prompt
        """
        # Model-specific optimizations
        optimized_prompt = prompt

        # For smaller models, simplify language
        if any(size in model_name.lower() for size in ["7b", "3b", "1b"]):
            # Simplify complex sentences
            optimized_prompt = optimized_prompt.replace(
                "Please utilize the appropriate tools",
                "Use the right tools"
            )
            optimized_prompt = optimized_prompt.replace(
                "comprehensive response",
                "complete answer"
            )

        # For code-focused models
        if any(code_term in model_name.lower() for code_term in ["code", "coder", "coding"]) and "tools" in optimized_prompt:
            optimized_prompt += "\n\nFocus on technical accuracy and provide code examples when relevant."

        # Token limit optimization
        if max_tokens:
            # Rough token estimation (1 token ≈ 4 characters)
            estimated_tokens = len(optimized_prompt) // 4
            if estimated_tokens > max_tokens * 0.8:  # Leave room for response
                # Truncate while preserving structure
                lines = optimized_prompt.split('\n')
                truncated_lines = []
                current_length = 0

                for line in lines:
                    if current_length + len(line) < max_tokens * 3:  # 3 chars per token estimate
                        truncated_lines.append(line)
                        current_length += len(line)
                    else:
                        break

                optimized_prompt = '\n'.join(truncated_lines)
                if len(truncated_lines) < len(lines):
                    optimized_prompt += "\n\n[Prompt truncated to fit token limit]"

        return optimized_prompt

    def create_multi_step_prompt(
        self,
        user_message: str,
        tools: list[MCPToolDefinition],
        steps: list[str] | None = None
    ) -> str:
        """
        Create a multi-step prompt for complex tasks requiring multiple tools.

        Args:
            user_message: User's request
            tools: Available MCP tools
            steps: Optional predefined steps

        Returns:
            Multi-step prompt
        """
        if not steps:
            # Auto-generate steps based on tools and message
            steps = self._generate_steps_from_context(user_message, tools)

        tool_list = self.generate_tool_list(tools, "detailed")

        prompt = f"""You need to complete a multi-step task. Follow these steps systematically:

Available tools:
{tool_list}

User request: {user_message}

Recommended approach:
"""
        
        for i, step in enumerate(steps, 1):
            prompt += f"{i}. {step}\n"

        prompt += """
Execute each step carefully, using the appropriate tools. Provide clear feedback after each step and summarize the final results."""

        return prompt

    def _generate_steps_from_context(
        self,
        user_message: str,
        tools: list[MCPToolDefinition]
    ) -> list[str]:
        """Generate steps based on user message and available tools."""
        message_lower = user_message.lower()
        steps = []

        # Analyze message for common patterns
        if "analyze" in message_lower or "data" in message_lower:
            steps.append("Gather and examine the relevant data")
            
        if "calculate" in message_lower or "compute" in message_lower:
            steps.append("Perform necessary calculations")
            
        if "search" in message_lower or "find" in message_lower:
            steps.append("Search for the requested information")
            
        if "send" in message_lower or "email" in message_lower or "notify" in message_lower:
            steps.append("Send notifications or communications")
            
        if "report" in message_lower or "summary" in message_lower:
            steps.append("Generate a comprehensive report or summary")

        # Default steps if none detected
        if not steps:
            steps = [
                "Understand the requirements",
                "Use appropriate tools to gather information",
                "Process and analyze the results",
                "Provide a comprehensive response"
            ]

        return steps

    def create_tool_chain_prompt(
        self,
        user_message: str,
        tool_chain: list[str],
        tools: list[MCPToolDefinition]
    ) -> str:
        """
        Create a prompt for executing tools in a specific sequence.

        Args:
            user_message: User's request
            tool_chain: Ordered list of tool names to execute
            tools: Available MCP tools

        Returns:
            Tool chain prompt
        """
        # Filter tools to only those in the chain
        chain_tools = []
        for tool_name in tool_chain:
            for tool in tools:
                if tool.name == tool_name:
                    chain_tools.append(tool)
                    break

        tool_descriptions = []
        for i, tool in enumerate(chain_tools, 1):
            tool_descriptions.append(f"{i}. **{tool.name}**: {tool.description}")

        prompt = f"""Execute the following tools in sequence to fulfill the user's request:

User request: {user_message}

Tool execution sequence:
{chr(10).join(tool_descriptions)}

Execute each tool in order, using the output from previous tools as input for subsequent ones when appropriate. Provide the final result after completing the entire chain."""

        return prompt

    def create_conditional_tool_prompt(
        self,
        user_message: str,
        tools: list[MCPToolDefinition],
        conditions: dict[str, str] | None = None
    ) -> str:
        """
        Create a prompt with conditional tool usage logic.

        Args:
            user_message: User's request
            tools: Available MCP tools
            conditions: Optional conditions for tool usage

        Returns:
            Conditional tool usage prompt
        """
        tool_list = self.generate_tool_list(tools, "simple")

        prompt = f"""You have access to these tools:
{tool_list}

User request: {user_message}

Use conditional logic to determine which tools to use:
- IF the request involves data analysis, use data-related tools
- IF the request involves communication, use messaging/email tools
- IF the request involves calculations, use mathematical tools
- IF the request involves file operations, use file system tools

"""

        if conditions:
            prompt += "Additional conditions:\n"
            for condition, action in conditions.items():
                prompt += f"- IF {condition}, THEN {action}\n"

        prompt += "\nAnalyze the request and use the appropriate tools based on these conditions."

        return prompt

    def create_error_recovery_prompt(
        self,
        user_message: str,
        failed_tool: str,
        error_message: str,
        alternative_tools: list[MCPToolDefinition]
    ) -> str:
        """
        Create a prompt for error recovery when a tool fails.

        Args:
            user_message: Original user request
            failed_tool: Name of the tool that failed
            error_message: Error message from the failed tool
            alternative_tools: Alternative tools that could be used

        Returns:
            Error recovery prompt
        """
        alt_tool_list = self.generate_tool_list(alternative_tools, "simple")

        prompt = f"""The tool '{failed_tool}' failed with error: {error_message}

Original request: {user_message}

Alternative approaches available:
{alt_tool_list}

Please try an alternative approach to fulfill the user's request. Consider:
1. Using a different tool that provides similar functionality
2. Breaking down the request into smaller parts
3. Modifying the approach based on the available tools

Proceed with the best alternative strategy."""

        return prompt

    def analyze_prompt_effectiveness(
        self,
        prompt: str,
        response: str,
        tools_used: list[str],
        success: bool
    ) -> dict[str, Any]:
        """
        Analyze the effectiveness of a prompt.

        Args:
            prompt: The prompt that was used
            response: The response received
            tools_used: List of tools that were used
            success: Whether the interaction was successful

        Returns:
            Analysis results
        """
        analysis = {
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tools_mentioned": self._count_tool_mentions(prompt),
            "tools_actually_used": len(tools_used),
            "tool_usage_rate": len(tools_used) / max(self._count_tool_mentions(prompt), 1),
            "success": success,
            "clarity_score": self._assess_prompt_clarity(prompt),
            "specificity_score": self._assess_prompt_specificity(prompt)
        }

        # Generate recommendations
        recommendations = []
        
        if analysis["tool_usage_rate"] < 0.5:
            recommendations.append("Consider making tool usage instructions more explicit")
            
        if analysis["clarity_score"] < 0.7:
            recommendations.append("Improve prompt clarity and structure")
            
        if analysis["specificity_score"] < 0.6:
            recommendations.append("Add more specific instructions and examples")

        analysis["recommendations"] = recommendations
        
        return analysis

    def _count_tool_mentions(self, prompt: str) -> int:
        """Count how many tools are mentioned in the prompt."""
        # Simple heuristic: count lines that look like tool descriptions
        lines = prompt.split('\n')
        tool_mentions = 0
        
        for line in lines:
            if line.strip().startswith('-') and ':' in line:
                tool_mentions += 1
                
        return tool_mentions

    def _assess_prompt_clarity(self, prompt: str) -> float:
        """Assess prompt clarity (0-1 scale)."""
        clarity_indicators = [
            "please" in prompt.lower(),
            "use" in prompt.lower(),
            ":" in prompt,  # Structured format
            len(prompt.split('\n')) > 3,  # Multi-line structure
            "request" in prompt.lower() or "task" in prompt.lower()
        ]
        
        return sum(clarity_indicators) / len(clarity_indicators)

    def _assess_prompt_specificity(self, prompt: str) -> float:
        """Assess prompt specificity (0-1 scale)."""
        specificity_indicators = [
            "step" in prompt.lower(),
            "if" in prompt.lower() or "when" in prompt.lower(),
            "example" in prompt.lower(),
            len([word for word in prompt.split() if word.isupper()]) > 0,  # Emphasis
            prompt.count('*') > 0 or prompt.count('**') > 0  # Markdown formatting
        ]
        
        return sum(specificity_indicators) / len(specificity_indicators)

    def get_template_suggestions(
        self,
        user_message: str,
        available_tools: list[MCPToolDefinition] | None = None
    ) -> list[str]:
        """
        Suggest appropriate templates based on user message and available tools.

        Args:
            user_message: User's message
            available_tools: Available MCP tools

        Returns:
            List of suggested template names
        """
        suggestions = []
        message_lower = user_message.lower()

        # Analyze message content for template suggestions
        if any(word in message_lower for word in ["how", "step", "guide", "tutorial"]):
            suggestions.append("step_by_step")

        if any(word in message_lower for word in ["problem", "issue", "solve", "fix"]):
            suggestions.append("problem_solving")

        if any(word in message_lower for word in ["use", "call", "execute", "run"]):
            suggestions.append("direct_tool")

        # Default to conversational if no specific patterns found
        if not suggestions:
            suggestions.append("conversational")

        # If many tools available, prefer structured approaches
        if available_tools and len(available_tools) > 3: # noqa
            if "conversational" in suggestions:
                suggestions.remove("conversational")
            if "step_by_step" not in suggestions:
                suggestions.insert(0, "step_by_step")

        return suggestions

    def list_templates(self) -> list[dict[str, Any]]:
        """
        List all available prompt templates.

        Returns:
            List of template information
        """
        return [
            {
                "name": template.name,
                "description": template.description,
                "strategy": template.strategy.value,
                "variables": template.variables,
                "tool_integration": template.tool_integration
            }
            for template in self.templates.values()
        ]


def demonstrate_prompt_engineering():
    """Demonstrate prompt engineering capabilities."""
    # Create prompt engineering instance
    prompt_eng = PromptEngineering()

    # Example tools
    example_tools = [
        MCPToolDefinition(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                },
                "required": ["expression"]
            },
            server_name="math-server"
        ),
        MCPToolDefinition(
            name="weather",
            description="Get weather information for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or location name"}
                },
                "required": ["location"]
            },
            server_name="weather-server"
        )
    ]

    # Generate different types of prompts
    user_message = "I need to calculate the area of a circle with radius 5 and also check the weather in New York"

    print("=== Direct Tool Prompt ===")
    direct_prompt = prompt_eng.generate_prompt("direct_tool", user_message, example_tools)
    print(direct_prompt)

    print("\n=== Conversational Prompt ===")
    conv_prompt = prompt_eng.generate_prompt("conversational", user_message, example_tools)
    print(conv_prompt)

    print("\n=== Step-by-Step Prompt ===")
    step_prompt = prompt_eng.generate_prompt("step_by_step", user_message, example_tools)
    print(step_prompt)

    print("\n=== Template Suggestions ===")
    suggestions = prompt_eng.get_template_suggestions(user_message, example_tools)
    print(f"Suggested templates: {suggestions}")


if __name__ == "__main__":
    demonstrate_prompt_engineering()
