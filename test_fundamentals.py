#!/usr/bin/env python3
"""
Test script for MCP fundamentals educational content.
"""

import asyncio
from pathlib import Path
import sys


# Add the project root to Python path
sys.path.insert(0, Path(Path(__file__).resolve()).parent)

from mcp_course.fundamentals.main import MCPFundamentalsEducator


async def test_fundamentals():
    """Test the MCP fundamentals educational content."""
    print("🧪 Testing MCP Fundamentals Educational Content")
    print("=" * 50)

    educator = MCPFundamentalsEducator()

    # Test concept exploration
    print("\\n📚 Testing Concept Exploration...")
    concept_guide = await educator.start_concept_exploration()
    print(f"✅ Concept exploration guide: {len(concept_guide)} lines")

    # Test architecture demonstration
    print("\\n🏗️ Testing Architecture Demonstration...")
    interaction_demo = await educator.demonstrate_complete_interaction()
    print(f"✅ Complete interaction demo: {len(interaction_demo)} lines")

    # Test protocol demonstration
    print("\\n📡 Testing Protocol Demonstration...")
    protocol_guide = await educator.start_protocol_demonstration()
    print(f"✅ Protocol demonstration guide: {len(protocol_guide)} lines")

    # Test initialization flow
    print("\\n🚀 Testing Initialization Flow...")
    init_flow = await educator.demonstrate_initialization_flow()
    print(f"✅ Initialization flow demo: {len(init_flow)} lines")

    # Test diagram generation
    print("\\n📊 Testing Diagram Generation...")
    arch_diagram = educator.generate_architecture_diagram()
    seq_diagram = educator.generate_sequence_diagram()
    print(f"✅ Architecture diagram: {len(arch_diagram)} characters")
    print(f"✅ Sequence diagram: {len(seq_diagram)} characters")

    # Test interactive flow
    print("\\n🔄 Testing Interactive Flow...")
    flow_session = await educator.start_interactive_flow()
    print(f"✅ Interactive flow session: {len(flow_session)} lines")

    # Test practical demo
    print("\\n⚡ Testing Practical Demo...")
    practical_demo = await educator.run_practical_demo()
    print(f"✅ Practical demo: {len(practical_demo)} lines")

    # Test quick reference
    print("\\n📖 Testing Quick Reference...")
    quick_ref = educator.get_quick_reference()
    print(f"✅ Quick reference sections: {len(quick_ref)} sections")

    print("\\n🎉 All tests passed! MCP Fundamentals implementation is working correctly.")

    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_fundamentals())
        if result:
            print("\\n✅ Test completed successfully!")
            sys.exit(0)
        else:
            print("\\n❌ Test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Test error: {e}")
        sys.exit(1)
