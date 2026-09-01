"""
Quick Test Script for AriadneMem
Validates that all imports and basic functionality work
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Core modules
        from main import AriadneMemSystem
        from models.memory_entry import Dialogue, MemoryEntry
        from core.ariadne_memory_builder import AriadneMemoryBuilder
        from core.ariadne_graph_retriever import AriadneGraphRetriever, GraphPath
        from core.ariadne_answer_generator import AriadneAnswerGenerator
        from database.vector_store import VectorStore
        from utils.llm_client import LLMClient
        from utils.embedding import EmbeddingModel
        print("Core modules imported successfully")
        
        # Test script modules
        from test_locomo10 import (
            QA, Turn, Session, Conversation, 
            LoCoMoSample, LoCoMoTester,
            calculate_metrics, aggregate_metrics
        )
        print("Test script modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test that config is properly set up"""
    print("\nTesting configuration...")
    
    try:
        import config
        
        # Check required config
        required = ['OPENAI_API_KEY', 'LLM_MODEL', 'EMBEDDING_MODEL']
        missing = []
        
        for key in required:
            if not hasattr(config, key):
                missing.append(key)
            else:
                value = getattr(config, key)
                if key == 'OPENAI_API_KEY':
                    if value == "your-api-key-here":
                        print(f"⚠ {key} needs to be configured (still has default value)")
                    else:
                        print(f"✓ {key} is configured")
                else:
                    print(f"✓ {key} = {value}")
        
        if missing:
            print(f"✗ Missing config keys: {missing}")
            return False
        
        return True
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False

def test_dataset():
    """Test that dataset file exists"""
    print("\nTesting dataset...")
    
    dataset_path = Path("dataset/locomo10.json")
    if dataset_path.exists():
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"✓ Dataset found: {dataset_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"✗ Dataset not found: {dataset_path}")
        return False

def test_system_init():
    """Test that AriadneMemSystem can be initialized"""
    print("\nTesting system initialization...")
    
    try:
        from main import AriadneMemSystem
        
        # Try to initialize (without clearing DB to be safe)
        print("  Initializing AriadneMemSystem...")
        system = AriadneMemSystem(clear_db=False)
        print("AriadneMemSystem initialized successfully")
        
        # Check components
        assert hasattr(system, 'llm_client'), "Missing llm_client"
        assert hasattr(system, 'embedding_model'), "Missing embedding_model"
        assert hasattr(system, 'vector_store'), "Missing vector_store"
        assert hasattr(system, 'memory_builder'), "Missing memory_builder"
        assert hasattr(system, 'graph_retriever'), "Missing graph_retriever"
        assert hasattr(system, 'answer_generator'), "Missing answer_generator"
        print("All system components present")
        
        return True
    except Exception as e:
        print(f"✗ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic add dialogue and retrieval"""
    print("\nTesting basic functionality...")
    
    try:
        from main import AriadneMemSystem
        from models.memory_entry import Dialogue
        
        # Initialize with clean DB
        print("  Creating test system...")
        system = AriadneMemSystem(clear_db=True)
        
        # Add a test dialogue
        print("  Adding test dialogue...")
        dialogue = Dialogue(
            dialogue_id=1,
            speaker="Alice",
            content="Let's meet at 2pm tomorrow",
            timestamp="2025-01-15T10:00:00"
        )
        system.add_dialogue(
            speaker=dialogue.speaker,
            content=dialogue.content,
            timestamp=dialogue.timestamp
        )
        system.finalize()
        print("Dialogue added successfully")
        
        # Try retrieval
        print("  Testing retrieval...")
        graph_path = system.graph_retriever.retrieve("When is the meeting?")
        print(f"Retrieval successful (retrieved {len(graph_path.nodes) if graph_path else 0} nodes)")
        
        # Try answer generation
        print("  Testing answer generation...")
        answer = system.answer_generator.generate_answer("When is the meeting?", graph_path)
        print(f"Answer generated: {answer[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("AriadneMem Quick Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Dataset", test_dataset()))
    results.append(("System Init", test_system_init()))
    
    # Only run functionality test if previous tests passed
    if all(r[1] for r in results):
        results.append(("Basic Functionality", test_basic_functionality()))
    else:
        print("\n⚠ Skipping functionality test due to previous failures")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! AriadneMem is ready to use.")
        print("\nNext step: Run the full test with:")
        print("  python test_locomo10.py --dataset dataset/locomo10.json --num-samples 1")
        return 0
    else:
        print("\nSome tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
