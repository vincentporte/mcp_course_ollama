"""
Visual diagram generation for MCP component relationships.

This module provides utilities to generate visual representations of MCP
architecture, component interactions, and protocol flows.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DiagramType(Enum):
    """Types of diagrams that can be generated."""
    ARCHITECTURE = "architecture"
    SEQUENCE = "sequence"
    COMPONENT = "component"
    FLOW = "flow"


@dataclass
class DiagramNode:
    """Represents a node in a diagram."""
    id: str
    label: str
    type: str
    properties: dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class DiagramEdge:
    """Represents an edge/connection in a diagram."""
    source: str
    target: str
    label: str = ""
    type: str = "default"
    properties: dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class MCPDiagramGenerator:
    """Generates visual diagrams for MCP concepts and architecture."""

    def __init__(self):
        self.nodes: list[DiagramNode] = []
        self.edges: list[DiagramEdge] = []

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.edges.clear()

    def add_node(self, node: DiagramNode) -> None:
        """Add a node to the diagram."""
        self.nodes.append(node)

    def add_edge(self, edge: DiagramEdge) -> None:
        """Add an edge to the diagram."""
        self.edges.append(edge)

    def generate_mcp_architecture_diagram(self) -> str:
        """Generate a comprehensive MCP architecture diagram."""
        self.clear()

        # Add core components
        self.add_node(DiagramNode("user", "User", "actor"))
        self.add_node(DiagramNode("llm", "Large Language Model", "system"))
        self.add_node(DiagramNode("client", "MCP Client", "component"))
        self.add_node(DiagramNode("server", "MCP Server", "component"))
        self.add_node(DiagramNode("tools", "Tools", "capability"))
        self.add_node(DiagramNode("resources", "Resources", "capability"))
        self.add_node(DiagramNode("prompts", "Prompts", "capability"))

        # Add connections
        self.add_edge(DiagramEdge("user", "llm", "Interacts with"))
        self.add_edge(DiagramEdge("llm", "client", "Uses capabilities via"))
        self.add_edge(DiagramEdge("client", "server", "Connects to", "bidirectional"))
        self.add_edge(DiagramEdge("server", "tools", "Provides"))
        self.add_edge(DiagramEdge("server", "resources", "Exposes"))
        self.add_edge(DiagramEdge("server", "prompts", "Offers"))

        return self._generate_mermaid_diagram(DiagramType.ARCHITECTURE)

    def generate_protocol_sequence_diagram(self) -> str:
        """Generate a sequence diagram showing MCP protocol flow."""
        sequence_steps = [
            ("Client", "Server", "initialize", "Establish connection"),
            ("Server", "Client", "initialize_response", "Confirm capabilities"),
            ("Client", "Server", "tools/list", "Request available tools"),
            ("Server", "Client", "tools/list_response", "Return tool definitions"),
            ("Client", "Server", "tools/call", "Execute specific tool"),
            ("Server", "Client", "tools/call_response", "Return execution result"),
            ("Client", "Server", "resources/read", "Access resource data"),
            ("Server", "Client", "resources/read_response", "Provide resource content")
        ]

        mermaid_lines = [
            "sequenceDiagram",
            "    participant C as MCP Client",
            "    participant S as MCP Server",
            "    participant T as Tools/Resources",
            ""
        ]

        for i, (source, target, method, description) in enumerate(sequence_steps, 1):
            if source == "Client":
                source_abbr = "C"
            elif source == "Server":
                source_abbr = "S"
            else:
                source_abbr = "T"

            if target == "Client":
                target_abbr = "C"
            elif target == "Server":
                target_abbr = "S"
            else:
                target_abbr = "T"

            mermaid_lines.append(f"    {source_abbr}->>+{target_abbr}: {i}. {method}")
            mermaid_lines.append(f"    Note over {source_abbr},{target_abbr}: {description}")

            if "response" in method:
                mermaid_lines.append(f"    {target_abbr}-->>-{source_abbr}: Success/Error")

            mermaid_lines.append("")

        return "\n".join(mermaid_lines)

    def generate_component_interaction_diagram(self) -> str:
        """Generate a diagram showing how MCP components interact."""
        mermaid_lines = [
            "graph TD",
            "    subgraph \"User Environment\"",
            "        U[👤 User] --> LLM[🤖 LLM]",
            "    end",
            "",
            "    subgraph \"MCP Ecosystem\"",
            "        LLM --> C[📱 MCP Client]",
            "        C <--> S1[🖥️ MCP Server 1]",
            "        C <--> S2[🖥️ MCP Server 2]",
            "        C <--> S3[🖥️ MCP Server N]",
            "    end",
            "",
            "    subgraph \"Server 1 Capabilities\"",
            "        S1 --> T1[🔧 File Tools]",
            "        S1 --> R1[📄 File Resources]",
            "    end",
            "",
            "    subgraph \"Server 2 Capabilities\"",
            "        S2 --> T2[🌐 Web Tools]",
            "        S2 --> R2[🔗 Web Resources]",
            "    end",
            "",
            "    subgraph \"Server N Capabilities\"",
            "        S3 --> T3[🗄️ Database Tools]",
            "        S3 --> R3[📊 Data Resources]",
            "    end",
            "",
            "    classDef user fill:#e1f5fe",
            "    classDef llm fill:#f3e5f5",
            "    classDef client fill:#e8f5e8",
            "    classDef server fill:#fff3e0",
            "    classDef capability fill:#fce4ec",
            "",
            "    class U user",
            "    class LLM llm",
            "    class C client",
            "    class S1,S2,S3 server",
            "    class T1,T2,T3,R1,R2,R3 capability"
        ]

        return "\n".join(mermaid_lines)

    def generate_tool_execution_flow(self) -> str:
        """Generate a flowchart showing tool execution process."""
        mermaid_lines = [
            "flowchart TD",
            "    Start([User Request]) --> Parse[LLM Parses Request]",
            "    Parse --> Decide{Need External Tool?}",
            "    Decide -->|No| Direct[Direct LLM Response]",
            "    Decide -->|Yes| Identify[Identify Required Tool]",
            "    Identify --> Check{Tool Available?}",
            "    Check -->|No| Error[Return Error Message]",
            "    Check -->|Yes| Call[Client Calls MCP Server]",
            "    Call --> Execute[Server Executes Tool]",
            "    Execute --> Validate{Execution Success?}",
            "    Validate -->|No| Error",
            "    Validate -->|Yes| Return[Return Results to Client]",
            "    Return --> Integrate[Client Integrates Results]",
            "    Integrate --> Response[LLM Generates Final Response]",
            "    Direct --> End([Response to User])",
            "    Response --> End",
            "    Error --> End",
            "",
            "    classDef process fill:#e3f2fd",
            "    classDef decision fill:#fff3e0",
            "    classDef error fill:#ffebee",
            "    classDef success fill:#e8f5e8",
            "",
            "    class Parse,Identify,Call,Execute,Integrate process",
            "    class Decide,Check,Validate decision",
            "    class Error error",
            "    class Return,Response,Direct success"
        ]

        return "\n".join(mermaid_lines)

    def generate_data_flow_diagram(self) -> str:
        """Generate a diagram showing data flow in MCP interactions."""
        mermaid_lines = [
            "graph LR",
            "    subgraph \"Input\"",
            "        UR[User Request]",
            "        UR --> UP[User Prompt]",
            "    end",
            "",
            "    subgraph \"Processing\"",
            "        UP --> LLM[LLM Analysis]",
            "        LLM --> TC[Tool Call Decision]",
            "        TC --> TR[Tool Request]",
            "    end",
            "",
            "    subgraph \"MCP Layer\"",
            "        TR --> MC[MCP Client]",
            "        MC --> MS[MCP Server]",
            "        MS --> TE[Tool Execution]",
            "        TE --> TD[Tool Data]",
            "    end",
            "",
            "    subgraph \"Response\"",
            "        TD --> MC2[Client Processing]",
            "        MC2 --> LR[LLM Integration]",
            "        LR --> FR[Final Response]",
            "    end",
            "",
            "    subgraph \"Output\"",
            "        FR --> UO[User Output]",
            "    end",
            "",
            "    classDef input fill:#e1f5fe",
            "    classDef processing fill:#f3e5f5",
            "    classDef mcp fill:#e8f5e8",
            "    classDef response fill:#fff3e0",
            "    classDef output fill:#fce4ec",
            "",
            "    class UR,UP input",
            "    class LLM,TC,TR processing",
            "    class MC,MS,TE,TD mcp",
            "    class MC2,LR,FR response",
            "    class UO output"
        ]

        return "\n".join(mermaid_lines)

    def _generate_mermaid_diagram(self, diagram_type: DiagramType) -> str:
        """Generate Mermaid diagram syntax from nodes and edges."""
        lines = ["graph TD"] if diagram_type == DiagramType.ARCHITECTURE else ["graph LR"]

        # Add nodes
        for node in self.nodes:
            node_style = self._get_node_style(node.type)
            lines.append(f"    {node.id}[{node_style}{node.label}]")

        # Add edges
        for edge in self.edges:
            arrow = "-->" if edge.type != "bidirectional" else "<-->"
            if edge.label:
                lines.append(f"    {edge.source} {arrow}|{edge.label}| {edge.target}")
            else:
                lines.append(f"    {edge.source} {arrow} {edge.target}")

        # Add styling
        lines.extend([
            "",
            "    classDef actor fill:#e1f5fe",
            "    classDef system fill:#f3e5f5",
            "    classDef component fill:#e8f5e8",
            "    classDef capability fill:#fff3e0"
        ])

        return "\n".join(lines)

    def _get_node_style(self, node_type: str) -> str:
        """Get appropriate node styling based on type."""
        styles = {
            "actor": "👤 ",
            "system": "🤖 ",
            "component": "📱 ",
            "capability": "🔧 "
        }
        return styles.get(node_type, "")

    def generate_all_diagrams(self) -> dict[str, str]:
        """Generate all available MCP diagrams."""
        return {
            "architecture": self.generate_mcp_architecture_diagram(),
            "sequence": self.generate_protocol_sequence_diagram(),
            "components": self.generate_component_interaction_diagram(),
            "tool_flow": self.generate_tool_execution_flow(),
            "data_flow": self.generate_data_flow_diagram()
        }

    def get_diagram_explanations(self) -> dict[str, str]:
        """Get explanations for each diagram type."""
        return {
            "architecture": "Shows the high-level MCP architecture with core components and their relationships",
            "sequence": "Illustrates the step-by-step protocol communication between client and server",
            "components": "Demonstrates how multiple MCP servers can provide different capabilities to a single client",
            "tool_flow": "Explains the decision-making process for tool execution in MCP interactions",
            "data_flow": "Traces how data moves through the MCP ecosystem from user input to final response"
        }


class InteractiveDiagramExplorer:
    """Provides interactive exploration of MCP diagrams with explanations."""

    def __init__(self):
        self.generator = MCPDiagramGenerator()
        self.current_diagram = None
        self.current_type = None

    def explore_architecture(self) -> tuple[str, list[str]]:
        """Explore MCP architecture with guided explanations."""
        diagram = self.generator.generate_mcp_architecture_diagram()
        explanations = [
            "🏗️ MCP Architecture Overview:",
            "",
            "This diagram shows the core components of the MCP ecosystem:",
            "",
            "👤 User: The person interacting with the AI system",
            "🤖 LLM: The language model that processes requests and generates responses",
            "📱 MCP Client: Bridges the LLM with external capabilities",
            "🖥️ MCP Server: Provides tools, resources, and prompts",
            "🔧 Tools: Executable functions that extend LLM capabilities",
            "📄 Resources: Structured data sources accessible to the LLM",
            "📝 Prompts: Reusable templates for generating responses",
            "",
            "Key Insights:",
            "• The client acts as a bridge between LLM and servers",
            "• Servers can provide multiple types of capabilities",
            "• The architecture is modular and extensible",
            "• Communication follows standardized protocols"
        ]

        self.current_diagram = diagram
        self.current_type = "architecture"
        return diagram, explanations

    def explore_protocol_flow(self) -> tuple[str, list[str]]:
        """Explore MCP protocol communication flow."""
        diagram = self.generator.generate_protocol_sequence_diagram()
        explanations = [
            "🔄 MCP Protocol Communication Flow:",
            "",
            "This sequence diagram shows the typical interaction pattern:",
            "",
            "1. Initialization: Client and server establish connection",
            "2. Capability Discovery: Client learns what server offers",
            "3. Tool Execution: Client requests specific tool operations",
            "4. Resource Access: Client reads data from server resources",
            "",
            "Protocol Features:",
            "• JSON-RPC based communication",
            "• Request-response pattern with unique IDs",
            "• Error handling and status reporting",
            "• Asynchronous operation support",
            "",
            "Benefits:",
            "• Standardized communication protocol",
            "• Language and platform agnostic",
            "• Extensible for custom implementations",
            "• Robust error handling mechanisms"
        ]

        self.current_diagram = diagram
        self.current_type = "sequence"
        return diagram, explanations

    def get_learning_checkpoints(self) -> list[str]:
        """Get key learning checkpoints for MCP concepts."""
        return [
            "✅ Understand the role of each MCP component",
            "✅ Recognize the client-server communication pattern",
            "✅ Identify how tools extend LLM capabilities",
            "✅ Appreciate the modular architecture benefits",
            "✅ Grasp the protocol communication flow",
            "✅ See how resources provide structured data access"
        ]
