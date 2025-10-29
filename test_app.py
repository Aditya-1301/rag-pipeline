#!/usr/bin/env python3
"""Quick test to verify app.py works correctly."""

import sys
import os

# Test imports
print("Testing app.py imports...")
try:
    from app import (
        check_if_embeddings_exist,
        DOCUMENT_CONFIG,
        VectorStore,
        MetadataStore,
        rag_query
    )
    print("✅ All imports successful!\n")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test configuration
print("Testing configuration...")
save_dir = DOCUMENT_CONFIG["save_dir"]
doc_path = DOCUMENT_CONFIG["sample_path"]
print(f"  Save directory: {save_dir}")
print(f"  Document path: {doc_path}")
print(f"  Document exists: {os.path.exists(doc_path)}")
print()

# Test embeddings existence
print("Testing embedding detection...")
exists = check_if_embeddings_exist(save_dir)
print(f"  Embeddings exist: {exists}")
print()

if exists:
    print("Testing vector store loading...")
    try:
        vs = VectorStore.load(save_dir, verbose=False)
        ms = MetadataStore.load(os.path.join(save_dir, "metadata.pkl"), verbose=False)
        print(f"✅ Loaded {vs.index.ntotal} vectors")
        print(f"✅ Loaded {len(ms.chunks)} metadata entries\n")
        
        # Test a simple query (without LLM, just retrieval)
        print("Testing retrieval...")
        from app import initialize_rag_demo
        initialize_rag_demo()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe app is ready to run. Start it with:")
        print("  python app.py")
        print("\nOr in Docker:")
        print("  docker-compose up")
        
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("⚠️  No embeddings found. On first run, the app will:")
    print("   1. Load the document")
    print("   2. Generate embeddings (takes ~5 minutes)")
    print("   3. Save them for future use")
    print("\nThis is normal for first-time setup!")
