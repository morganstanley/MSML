"""
AriadneMem - Structured Memory System for Lifelong LLM Agents
Main system class integrating all components

Paper: AriadneMem: Threading the Maze of Lifelong Memory for LLM Agents
"""
from typing import List, Optional
from models.memory_entry import Dialogue, MemoryEntry
from utils.llm_client import LLMClient
from utils.embedding import EmbeddingModel
from database.vector_store import VectorStore
from core.ariadne_memory_builder import AriadneMemoryBuilder
from core.ariadne_graph_retriever import AriadneGraphRetriever, GraphPath
from core.ariadne_answer_generator import AriadneAnswerGenerator
import config


class AriadneMemSystem:
    """
    AriadneMem Main System
    
    Two-phase pipeline for lifelong memory:
    Phase I (Offline): Asynchronous Memory Construction
        - Entropy-aware gating to filter low-information inputs
        - Conflict-aware coarsening to merge duplicates while preserving state updates
    Phase II (Online): Real-Time Structural Reasoning
        - Hybrid retrieval (semantic + lexical)
        - Algorithmic bridge discovery (Steiner tree approximation)
        - Topology-aware synthesis
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        clear_db: bool = False,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
        redundancy_threshold: Optional[float] = None,
        coarsening_threshold: Optional[float] = None,
        builder_model: Optional[str] = None,
        answer_model: Optional[str] = None,
        reasoning_mode: Optional[str] = None
    ):
        """
        Initialize AriadneMem system

        Args:
        - api_key: OpenAI API key
        - model: Default LLM model name (for all phases)
        - base_url: Custom OpenAI base URL (for compatible APIs)
        - db_path: Database path
        - table_name: Memory table name
        - clear_db: Whether to clear existing database
        - enable_thinking: Enable deep thinking mode (for Qwen and compatible models)
        - use_streaming: Enable streaming responses
        - redundancy_threshold: Threshold for entropy-aware gating (lambda_red in paper)
        - coarsening_threshold: Threshold for conflict-aware coarsening (lambda_coal in paper)
        - builder_model: Phase I model override (extraction + coarsening). Falls back to model/config.
        - answer_model: Phase II model override (topology-aware synthesis). Falls back to model/config.
        - reasoning_mode: Reasoning mode ("eco", "pro", "custom"). Falls back to config.REASONING_MODE.
        
        Note: Parallel processing settings are read from config.py:
        - ENABLE_PARALLEL_PROCESSING
        - MAX_PARALLEL_WORKERS
        """
        print("=" * 60)
        print("Initializing AriadneMem System")
        print("=" * 60)

        # Apply reasoning mode override (affects config globals for this session)
        effective_mode = reasoning_mode or getattr(config, 'REASONING_MODE', 'eco')
        if reasoning_mode and reasoning_mode in config.MODE_CONFIGS:
            mode_cfg = config.MODE_CONFIGS[reasoning_mode]
            config.MAX_REASONING_PATH_DEPTH = mode_cfg["MAX_REASONING_PATH_DEPTH"]
            config.MAX_REASONING_PATHS = mode_cfg["MAX_REASONING_PATHS"]
            config.REASONING_MODE = reasoning_mode
            # Update prompt template based on mode
            if reasoning_mode == "pro":
                config.ANSWER_USER_PROMPT_TEMPLATE = config._PRO_USER_PROMPT_TEMPLATE
            elif reasoning_mode == "custom" and getattr(config, '_CUSTOM_USER_PROMPT_TEMPLATE', None):
                config.ANSWER_USER_PROMPT_TEMPLATE = config._CUSTOM_USER_PROMPT_TEMPLATE
            else:
                config.ANSWER_USER_PROMPT_TEMPLATE = config._ECO_USER_PROMPT_TEMPLATE
        print(f"  Reasoning mode: {effective_mode}")

        # Initialize core components
        # Default LLM client (used unless per-component overrides are set)
        default_model = model or getattr(config, 'LLM_MODEL', 'gpt-4o')
        self.llm_client = LLMClient(
            api_key=api_key,
            model=default_model,
            base_url=base_url,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming
        )

        # Per-component LLM clients
        # Priority: __init__ param > config.py > default_model
        effective_builder_model = builder_model or getattr(config, 'BUILDER_LLM_MODEL', None)
        effective_answer_model = answer_model or getattr(config, 'ANSWER_LLM_MODEL', None)

        if effective_builder_model and effective_builder_model != default_model:
            print(f"  Phase I  (Builder) model: {effective_builder_model}")
            self.builder_llm_client = LLMClient(
                api_key=api_key, model=effective_builder_model,
                base_url=base_url, enable_thinking=enable_thinking,
                use_streaming=use_streaming
            )
        else:
            self.builder_llm_client = self.llm_client

        if effective_answer_model and effective_answer_model != default_model:
            print(f"  Phase II (Answer)  model: {effective_answer_model}")
            self.answer_llm_client = LLMClient(
                api_key=api_key, model=effective_answer_model,
                base_url=base_url, enable_thinking=enable_thinking,
                use_streaming=use_streaming
            )
        else:
            self.answer_llm_client = self.llm_client

        print(f"  Default model: {default_model}")

        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(
            db_path=db_path,
            embedding_model=self.embedding_model,
            table_name=table_name
        )

        if clear_db:
            print("\nClearing existing database...")
            self.vector_store.clear()

        # Initialize pipeline modules
        # Phase I: Memory Construction (entropy-aware gating + conflict-aware coarsening)
        self.memory_builder = AriadneMemoryBuilder(
            llm_client=self.builder_llm_client,
            vector_store=self.vector_store,
            redundancy_threshold=redundancy_threshold or getattr(config, 'REDUNDANCY_THRESHOLD', 0.92),
            coarsening_threshold=coarsening_threshold or getattr(config, 'COARSENING_THRESHOLD', 0.96)
        )

        # Phase II: Structural Reasoning (bridge discovery + topology-aware synthesis)
        self.graph_retriever = AriadneGraphRetriever(
            llm_client=self.llm_client,
            vector_store=self.vector_store
        )

        self.answer_generator = AriadneAnswerGenerator(
            llm_client=self.answer_llm_client
        )

        print("\nSystem initialization complete!")
        print("=" * 60)

    def add_dialogue(self, speaker: str, content: str, timestamp: Optional[str] = None):
        """
        Add a single dialogue

        Args:
        - speaker: Speaker name
        - content: Dialogue content
        - timestamp: Timestamp (ISO 8601 format)
        """
        dialogue_id = self.memory_builder.processed_count + len(self.memory_builder.dialogue_buffer) + 1
        dialogue = Dialogue(
            dialogue_id=dialogue_id,
            speaker=speaker,
            content=content,
            timestamp=timestamp
        )
        self.memory_builder.add_dialogue(dialogue)

    def add_dialogues(self, dialogues: List[Dialogue]):
        """
        Batch add dialogues

        Args:
        - dialogues: List of dialogues (dialogue_id auto-assigned if 0)
        """
        # Auto-assign dialogue_id for dialogues that don't have one
        base_id = self.memory_builder.processed_count + len(self.memory_builder.dialogue_buffer) + 1
        for idx, d in enumerate(dialogues):
            if d.dialogue_id == 0:
                d.dialogue_id = base_id + idx
        self.memory_builder.add_dialogues(dialogues)

    def finalize(self):
        """
        Finalize dialogue input, process any remaining buffer
        Also builds enhanced index for better performance
        """
        self.memory_builder.process_remaining()
        
        # Build enhanced index (aggregations, relations, temporal index)
        self.memory_builder.build_enhanced_index()
        
        # Load enhanced index into retriever
        enhanced_index = self.vector_store.load_enhanced_index()
        if enhanced_index:
            self.graph_retriever.set_enhanced_index(enhanced_index)

    def ask(self, question: str, top_k: int = 5) -> str:
        """
        Ask question - Core Q&A interface (Phase II: Online Reasoning)

        Args:
        - question: User question
        - top_k: Number of top results for hybrid retrieval

        Returns:
        - Answer
        """
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        # Phase II Step 1: Structural retrieval (hybrid recall + bridge discovery)
        # top_k is controlled by config (SEMANTIC_TOP_K, KEYWORD_TOP_K, STRUCTURED_TOP_K)
        graph_path = self.graph_retriever.retrieve(question)

        # Phase II Step 2: Topology-aware synthesis
        answer = self.answer_generator.generate_answer(question, graph_path)

        print("\nAnswer:")
        print(answer)
        print("=" * 60 + "\n")

        return answer

    def get_all_memories(self) -> List[MemoryEntry]:
        """
        Get all memory entries (for debugging)
        """
        return self.vector_store.get_all_entries()

    def print_memories(self):
        """
        Print all memory entries (for debugging)
        """
        memories = self.get_all_memories()
        print("\n" + "=" * 60)
        print(f"All Memory Entries ({len(memories)} total)")
        print("=" * 60)

        for i, memory in enumerate(memories, 1):
            print(f"\n[Entry {i}]")
            print(f"ID: {memory.entry_id}")
            print(f"Restatement: {memory.lossless_restatement}")
            if memory.timestamp:
                print(f"Time: {memory.timestamp}")
            if memory.location:
                print(f"Location: {memory.location}")
            if memory.persons:
                print(f"Persons: {', '.join(memory.persons)}")
            if memory.entities:
                print(f"Entities: {', '.join(memory.entities)}")
            if memory.topic:
                print(f"Topic: {memory.topic}")
            print(f"Keywords: {', '.join(memory.keywords)}")

        print("\n" + "=" * 60)


# Convenience function
def create_system(
    clear_db: bool = False,
    redundancy_threshold: Optional[float] = None,
    coarsening_threshold: Optional[float] = None,
    builder_model: Optional[str] = None,
    answer_model: Optional[str] = None,
    reasoning_mode: Optional[str] = None
) -> AriadneMemSystem:
    """
    Create AriadneMem system instance (uses config.py defaults when None)
    
    Args:
    - clear_db: Clear existing database on init
    - redundancy_threshold: Entropy-aware gating threshold (lambda_red)
    - coarsening_threshold: Conflict-aware coarsening threshold (lambda_coal)
    - builder_model: Phase I model override
    - answer_model: Phase II model override
    - reasoning_mode: "eco", "pro", or "custom"
    """
    return AriadneMemSystem(
        clear_db=clear_db,
        redundancy_threshold=redundancy_threshold,
        coarsening_threshold=coarsening_threshold,
        builder_model=builder_model,
        answer_model=answer_model,
        reasoning_mode=reasoning_mode
    )


if __name__ == "__main__":
    # Quick test
    print("Running AriadneMem Quick Test...")

    system = create_system(clear_db=True)
    print(f"Using embedding model: {system.memory_builder.vector_store.embedding_model.model_name}")
    print(f"Model type: {system.memory_builder.vector_store.embedding_model.model_type}")

    # Add some test dialogues
    system.add_dialogue("Alice", "Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product", "2025-11-15T14:30:00")
    system.add_dialogue("Bob", "Okay, I'll prepare the materials", "2025-11-15T14:31:00")
    system.add_dialogue("Alice", "Remember to bring the market research report from last time", "2025-11-15T14:32:00")

    # Finalize input
    system.finalize()

    # View memories
    system.print_memories()

    # Ask questions
    print("\nTesting structural retrieval...")
    system.ask("What time (in hour) will Alice and Bob meet?")
    
    print("\nTesting adversarial question...")
    question = "What is Alice's favorite food?"
    graph_path = system.graph_retriever.retrieve(question)
    answer = system.answer_generator.generate_answer(question, graph_path)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    print("\nQuick test completed!")
