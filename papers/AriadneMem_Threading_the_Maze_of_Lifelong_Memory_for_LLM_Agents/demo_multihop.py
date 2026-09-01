#!/usr/bin/env python3
"""
Multi-Hop Path Reasoning Demo Script
Demonstrates how AriadneMem automatically discovers and utilizes multi-hop reasoning paths

Paper Reference: Section 2.3 - Multi-Hop Path Mining (Eq. 10)
"""

from main import AriadneMemSystem
from models.memory_entry import Dialogue
from datetime import datetime, timedelta

def demo_basic_multihop():
    """Demonstrate basic multi-hop reasoning"""
    print("=" * 80)
    print("Demo 1: Basic Multi-Hop Reasoning (A -> B -> C)")
    print("=" * 80)
    
    system = AriadneMemSystem()
    
    # Scenario: Plan change
    base_time = datetime.now()
    dialogues = [
        Dialogue(
            dialogue_id=1,
            speaker="User",
            content="Alice plans to visit Paris next week for a conference.",
            timestamp=(base_time + timedelta(days=1)).isoformat()
        ),
        Dialogue(
            dialogue_id=2,
            speaker="Assistant",
            content="That sounds exciting! When is the conference?",
            timestamp=(base_time + timedelta(days=1, hours=1)).isoformat()
        ),
        Dialogue(
            dialogue_id=3,
            speaker="User",
            content="The Paris flight was cancelled due to bad weather.",
            timestamp=(base_time + timedelta(days=5)).isoformat()
        ),
        Dialogue(
            dialogue_id=4,
            speaker="User",
            content="Alice booked a flight to London instead.",
            timestamp=(base_time + timedelta(days=6)).isoformat()
        ),
        Dialogue(
            dialogue_id=5,
            speaker="User",
            content="The London conference will be held at the Royal Hotel.",
            timestamp=(base_time + timedelta(days=7)).isoformat()
        )
    ]
    
    print("\nAdding dialogue history...")
    for d in dialogues:
        system.add_dialogue(d.speaker, d.content, d.timestamp)
    
    print("Memory construction complete")
    system.finalize()
    
    # Test multi-hop reasoning
    print("\n" + "=" * 80)
    print("Testing Multi-Hop Reasoning Question")
    print("=" * 80)
    
    question = "Where is Alice going for the conference?"
    print(f"\nQuestion: {question}")
    
    # Check reasoning paths
    graph_path = system.graph_retriever.retrieve(question)
    
    print(f"\nDiscovered {len(graph_path.reasoning_paths)} reasoning paths:")
    for i, path in enumerate(graph_path.reasoning_paths, 1):
        print(f"\n  Path {i}: ({len(path)} hops)")
        for j, node in enumerate(path):
            arrow = "    -> " if j > 0 else "    "
            timestamp_str = node.timestamp[:16] if node.timestamp else "Unknown"
            content = node.lossless_restatement[:60] + "..." if len(node.lossless_restatement) > 60 else node.lossless_restatement
            print(f"{arrow}[{timestamp_str}] {content}")
    
    # Generate answer
    print("\n" + "-" * 80)
    answer = system.ask(question)
    print(f"Answer: {answer}")
    print("=" * 80)


def demo_causal_reasoning():
    """Demonstrate causal reasoning"""
    print("\n\n" + "=" * 80)
    print("Demo 2: Causal Reasoning (Why did X cause Y?)")
    print("=" * 80)
    
    system = AriadneMemSystem()
    
    # Scenario: Project failure causal chain
    base_time = datetime.now()
    dialogues = [
        Dialogue(
            dialogue_id=1,
            speaker="User",
            content="Bob started working on Project Alpha.",
            timestamp=(base_time + timedelta(days=1)).isoformat()
        ),
        Dialogue(
            dialogue_id=2,
            speaker="User",
            content="Bob missed the first milestone because the requirements were unclear.",
            timestamp=(base_time + timedelta(days=10)).isoformat()
        ),
        Dialogue(
            dialogue_id=3,
            speaker="User",
            content="The team lead was not happy with the delay.",
            timestamp=(base_time + timedelta(days=11)).isoformat()
        ),
        Dialogue(
            dialogue_id=4,
            speaker="User",
            content="Bob was assigned additional resources to catch up.",
            timestamp=(base_time + timedelta(days=12)).isoformat()
        ),
        Dialogue(
            dialogue_id=5,
            speaker="User",
            content="Despite the extra help, Project Alpha was cancelled.",
            timestamp=(base_time + timedelta(days=20)).isoformat()
        )
    ]
    
    print("\nAdding dialogue history...")
    for d in dialogues:
        system.add_dialogue(d.speaker, d.content, d.timestamp)
    
    print("Memory construction complete")
    system.finalize()
    
    # Test causal reasoning
    print("\n" + "=" * 80)
    print("Testing Causal Reasoning Question")
    print("=" * 80)
    
    question = "Why was Project Alpha cancelled?"
    print(f"\nQuestion: {question}")
    
    # Check reasoning paths
    graph_path = system.graph_retriever.retrieve(question)
    
    print(f"\nDiscovered {len(graph_path.reasoning_paths)} reasoning paths:")
    for i, path in enumerate(graph_path.reasoning_paths, 1):
        if i <= 3:  # Show only first 3 paths
            print(f"\n  Path {i}: ({len(path)} hops)")
            for j, node in enumerate(path):
                arrow = "    -> " if j > 0 else "    "
                timestamp_str = node.timestamp[:16] if node.timestamp else "Unknown"
                content = node.lossless_restatement[:60] + "..." if len(node.lossless_restatement) > 60 else node.lossless_restatement
                print(f"{arrow}[{timestamp_str}] {content}")
    
    # Generate answer
    print("\n" + "-" * 80)
    answer = system.ask(question)
    print(f"Answer: {answer}")
    print("=" * 80)


def demo_path_statistics():
    """Show path statistics"""
    print("\n\n" + "=" * 80)
    print("Demo 3: Reasoning Path Statistics")
    print("=" * 80)
    
    system = AriadneMemSystem()
    
    # Create a more complex scenario
    base_time = datetime.now()
    dialogues = [
        Dialogue(
            dialogue_id=1,
            speaker="User",
            content="Charlie joined the company as a software engineer.",
            timestamp=(base_time + timedelta(days=1)).isoformat()
        ),
        Dialogue(
            dialogue_id=2,
            speaker="User",
            content="Charlie completed the onboarding training.",
            timestamp=(base_time + timedelta(days=5)).isoformat()
        ),
        Dialogue(
            dialogue_id=3,
            speaker="User",
            content="Charlie was assigned to the frontend team.",
            timestamp=(base_time + timedelta(days=7)).isoformat()
        ),
        Dialogue(
            dialogue_id=4,
            speaker="User",
            content="The frontend team was working on a new dashboard.",
            timestamp=(base_time + timedelta(days=8)).isoformat()
        ),
        Dialogue(
            dialogue_id=5,
            speaker="User",
            content="Charlie implemented the user authentication module.",
            timestamp=(base_time + timedelta(days=15)).isoformat()
        ),
        Dialogue(
            dialogue_id=6,
            speaker="User",
            content="Charlie was promoted to senior engineer after 6 months.",
            timestamp=(base_time + timedelta(days=180)).isoformat()
        ),
    ]
    
    print("\nAdding dialogue history...")
    for d in dialogues:
        system.add_dialogue(d.speaker, d.content, d.timestamp)
    
    print("Memory construction complete")
    system.finalize()
    
    # Analyze reasoning paths
    question = "What has Charlie accomplished since joining?"
    print(f"\nQuestion: {question}")
    
    graph_path = system.graph_retriever.retrieve(question)
    
    print(f"\nReasoning Path Statistics:")
    print(f"  - Total nodes: {len(graph_path.nodes)}")
    print(f"  - Total edges: {len(graph_path.edges)}")
    print(f"  - Discovered paths: {len(graph_path.reasoning_paths)}")
    
    if graph_path.reasoning_paths:
        path_lengths = [len(p) for p in graph_path.reasoning_paths]
        print(f"  - Average path length: {sum(path_lengths) / len(path_lengths):.1f} hops")
        print(f"  - Longest path: {max(path_lengths)} hops")
        print(f"  - Shortest path: {min(path_lengths)} hops")
        
        print("\n  Path details:")
        for i, path in enumerate(graph_path.reasoning_paths, 1):
            path_str = " -> ".join([f"Node{j+1}" for j in range(len(path))])
            print(f"    Path {i}: {path_str} ({len(path)} hops)")
    
    # Generate answer
    print("\n" + "-" * 80)
    answer = system.ask(question)
    print(f"Answer: {answer}")
    print("=" * 80)


def demo_comparison():
    """Compare with and without multi-hop reasoning"""
    print("\n\n" + "=" * 80)
    print("Demo 4: Value of Multi-Hop Reasoning")
    print("=" * 80)
    
    print("\nComparison:")
    print("  - AriadneMem: Automatically uses multi-hop path reasoning")
    print("  - Traditional methods: Provide discrete nodes, let LLM reason by itself")
    print()
    
    system = AriadneMemSystem()
    
    # Create a scenario requiring multi-hop reasoning
    base_time = datetime.now()
    dialogues = [
        Dialogue(
            dialogue_id=1,
            speaker="User",
            content="David bought a lottery ticket.",
            timestamp=(base_time + timedelta(days=1)).isoformat()
        ),
        Dialogue(
            dialogue_id=2,
            speaker="User",
            content="David won $1000 in the lottery.",
            timestamp=(base_time + timedelta(days=5)).isoformat()
        ),
        Dialogue(
            dialogue_id=3,
            speaker="User",
            content="David used the money to pay off his debt.",
            timestamp=(base_time + timedelta(days=6)).isoformat()
        ),
        Dialogue(
            dialogue_id=4,
            speaker="User",
            content="David started saving money for a new car.",
            timestamp=(base_time + timedelta(days=10)).isoformat()
        ),
    ]
    
    print("Adding dialogue history...")
    for d in dialogues:
        system.add_dialogue(d.speaker, d.content, d.timestamp)
    
    system.finalize()
    
    question = "How did David get money for his car savings?"
    print(f"\nComplex question: {question}")
    print("  (Requires connecting: buy ticket -> win lottery -> pay debt -> save money)")
    
    # Show how AriadneMem handles this
    graph_path = system.graph_retriever.retrieve(question)
    
    print(f"\nAriadneMem automatically discovers reasoning chain:")
    if graph_path.reasoning_paths:
        path = graph_path.reasoning_paths[0]  # Take first path
        for j, node in enumerate(path):
            arrow = "   -> " if j > 0 else "   "
            content = node.lossless_restatement[:50] + "..." if len(node.lossless_restatement) > 50 else node.lossless_restatement
            print(f"{arrow}{content}")
    
    print("\nAdvantages:")
    print("  - 0 extra LLM calls (pure graph algorithm)")
    print("  - Reasoning path visualization (explainability)")
    print("  - Reduced LLM reasoning burden (fewer errors)")
    
    # Generate answer
    print("\n" + "-" * 80)
    answer = system.ask(question)
    print(f"Final answer: {answer}")
    print("=" * 80)


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("AriadneMem Multi-Hop Path Reasoning Demonstration")
    print("Paper: Section 2.3 - Multi-Hop Path Mining (Eq. 10)")
    print("=" * 80)
    
    try:
        # Run all demos
        demo_basic_multihop()
        demo_causal_reasoning()
        demo_path_statistics()
        demo_comparison()
        
        print("\n\n" + "=" * 80)
        print("All demos completed!")
        print("=" * 80)
        print("\nKey takeaways:")
        print("  1. Multi-hop reasoning is automatic, no configuration needed")
        print("  2. Uses DFS graph algorithm, 0 extra LLM calls")
        print("  3. Reasoning paths are visualized for debugging")
        print("  4. Significantly improves accuracy on complex questions (+7-13%)")
        print()
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()
