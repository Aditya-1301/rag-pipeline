import os
import socket
import gradio as gr
import numpy as np
import traceback
from rag_utils import (
    DOCUMENT_CONFIG,
    TOKEN_CONFIG,
    EMBEDDING_DIM,
    DEFAULT_OLLAMA_MODEL,
    _demo_state,
    load_pdf,
    load_text,
    load_docx,
    chunk_pdf_pages,
    chunk_document_semantic,
    generate_embeddings,
    execute_query,
    check_if_embeddings_exist,
    DocumentChunk,
    MetadataStore,
    VectorStore
)


# ===== GRADIO DEMO INTERFACE WITH MULTI-DOCUMENT UPLOAD =====

# Global state for multi-document handling
_upload_state = {
    "documents": [],  # List of (filename, file_path) tuples
    "processing": False
}


def is_port_available(port: int) -> bool:
    """Check if a port is available for use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            return True
    except OSError:
        return False


def find_available_port(start_port: int = 7860, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        if is_port_available(port):
            return port
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}")


def process_uploaded_documents(files) -> str:
    """
    Process multiple uploaded documents.
    Generates embeddings for all documents and stores them in the vector store.
    
    Args:
        files: List of file paths (from Gradio file_count="multiple")
    """
    # Handle both single file and multiple files
    if files is None:
        return "❌ Error: No files uploaded. Please upload at least one PDF or document."
    
    # Ensure files is always a list
    if isinstance(files, str):
        files = [files]
    elif not isinstance(files, list):
        files = [files]
    
    # Filter out None values
    uploaded_files = [f for f in files if f is not None]
    
    if not uploaded_files:
        return "❌ Error: No files uploaded. Please upload at least one PDF or document."
    
    try:
        _upload_state["processing"] = True
        
        print(f"\n📚 Processing {len(uploaded_files)} document(s)...")
        print("=" * 70)
        
        # Initialize metadata store if not exists
        ms = MetadataStore()
        vs = VectorStore(EMBEDDING_DIM)
        
        total_chunks = 0
        
        # Process each uploaded document
        for file_path in uploaded_files:
            # Ensure file_path is a string
            if not isinstance(file_path, str):
                file_path = str(file_path)
            
            # Strip any whitespace or special characters
            file_path = file_path.strip().strip("[]'\"")
            
            if not file_path:
                continue
            
            filename = os.path.basename(file_path)
            print(f"\n📄 Processing: {filename}")
            
            try:
                # Load document based on file type
                if filename.lower().endswith('.pdf'):
                    # load_pdf expects a file path
                    document_data = load_pdf(file_path)
                    if not document_data:
                        print(f"   ⚠️  No content extracted from PDF")
                        continue
                    chunks_data = chunk_pdf_pages(
                        document_data,
                        chunk_size=DOCUMENT_CONFIG["chunk_size"],
                        overlap=DOCUMENT_CONFIG["chunk_overlap"]
                    )
                elif filename.lower().endswith(('.txt', '.md')):
                    # load_text handles file paths
                    document_text = load_text(file_path)
                    if not document_text or not document_text.strip():
                        print(f"   ⚠️  No content extracted from text file")
                        continue
                    # For text files, wrap in list of tuples with None as page_num
                    chunks_data = chunk_document_semantic(
                        document_text,
                        chunk_size=DOCUMENT_CONFIG["chunk_size"],
                        overlap=DOCUMENT_CONFIG["chunk_overlap"],
                        page_number=None
                    )
                elif filename.lower().endswith('.docx'):
                    # Load DOCX file
                    document_text = load_docx(file_path)
                    if not document_text or not document_text.strip():
                        print(f"   ⚠️  No content extracted from DOCX file")
                        continue
                    chunks_data = chunk_document_semantic(
                        document_text,
                        chunk_size=DOCUMENT_CONFIG["chunk_size"],
                        overlap=DOCUMENT_CONFIG["chunk_overlap"],
                        page_number=None
                    )
                else:
                    print(f"   ⚠️  Unsupported file type: {filename}")
                    continue
                
                if not chunks_data:
                    print(f"   ⚠️  No chunks created from {filename}")
                    continue
                
                # Create chunks with metadata
                document_chunks = []
                for chunk_text, page_num in chunks_data:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        source=filename,
                        page_number=page_num
                    )
                    ms.add_chunk(chunk)
                    document_chunks.append(chunk)
                
                print(f"   ✓ Created {len(document_chunks)} chunks")
                total_chunks += len(document_chunks)
                
                # Generate embeddings
                print(f"   ⏳ Generating embeddings...")
                generate_embeddings(document_chunks, batch_size=100, use_optimizations=True)
                print(f"   ✓ Embeddings generated")
                
                # Add to vector store
                embeddings = np.array([chunk.embedding for chunk in document_chunks])
                chunk_ids = [chunk.chunk_id for chunk in document_chunks]
                vs.add_vectors(embeddings, chunk_ids)
                
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if total_chunks == 0:
            return "❌ Error: No chunks were created from the uploaded documents. Please check the files contain readable text."
        
        # Save to disk
        SAVE_DIR = DOCUMENT_CONFIG["save_dir"]
        os.makedirs(SAVE_DIR, exist_ok=True)
        vs.save(SAVE_DIR)
        ms.save(os.path.join(SAVE_DIR, "metadata.pkl"))
        
        # Update demo state
        _demo_state["vector_store"] = vs
        _demo_state["metadata_store"] = ms
        _demo_state["initialized"] = True
        
        _upload_state["processing"] = False
        
        success_msg = f"""
✅ **Documents Processed Successfully!**

- **Documents:** {len(uploaded_files)}
- **Total Chunks:** {total_chunks}
- **Vector Store:** Saved to `{SAVE_DIR}`

You can now ask questions about your uploaded documents using the search box below.
        """
        
        print(f"\n{success_msg}")
        return success_msg
        
    except Exception as e:
        _upload_state["processing"] = False
        error_msg = f"❌ Error processing documents: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

def rag_demo_interface(query: str, top_k: int = 5, model: str = None) -> str:
    """
    Gradio-compatible RAG interface.
    Takes user query and returns formatted answer with sources.
    """
    try:
        # Check if documents have been processed
        if not _demo_state.get("initialized"):
            return "❌ Error: Please upload and process documents first using the 'Upload Documents' tab."
        
        # Use default model if not specified
        if model is None or model.strip() == "":
            model = DEFAULT_OLLAMA_MODEL
        
        # Execute query with user-specified parameters
        vs = _demo_state["vector_store"]
        ms = _demo_state["metadata_store"]
        result = execute_query(query, vs, ms, model_name=model, top_k=top_k, verbose=False)
        
        return result["formatted_answer"]
    
    except ValueError as e:
        # RAG system not initialized
        return f"❌ Error: {str(e)}\n\nPlease initialize the system first by uploading documents."
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"


def create_gradio_demo():
    """
    Create and configure Gradio interface for multi-document RAG system.
    Returns the Gradio Blocks interface with document upload and Q&A tabs.
    """
    
    with gr.Blocks(
        title="📚 RAG Q&A Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gr-box { border-radius: 12px; }
        .gr-button { border-radius: 8px; }
        .gr-textbox { border-radius: 8px; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 📚 RAG Q&A Assistant with Multi-Document Support
        
        Upload your documents and ask questions using semantic search + local LLM.
        **Powered by:** FAISS + Ollama + HuggingFace Embeddings
        """)
        
        with gr.Tabs():
            # ===== TAB 1: DOCUMENT UPLOAD =====
            with gr.Tab("📤 Upload Documents"):
                gr.Markdown("""
                ## Upload Your Documents
                
                Upload one or more documents (PDF, DOCX, TXT, MD) to index and search.
                The system will:
                1. Process each document
                2. Split into semantic chunks
                3. Generate embeddings
                4. Save to vector store for future queries
                """)
                
                with gr.Row():
                    file_upload = gr.File(
                        label="Upload Documents",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md", ".docx"],
                        type="filepath"
                    )
                
                process_btn = gr.Button("⚙️ Process Documents", variant="primary", size="lg")
                upload_status = gr.Markdown("*No documents uploaded yet*")
                
                # Connect upload button
                process_btn.click(
                    fn=process_uploaded_documents,
                    inputs=[file_upload],
                    outputs=upload_status
                )
            
            # ===== TAB 2: Q&A INTERFACE =====
            with gr.Tab("🔍 Ask Questions"):
                gr.Markdown("""
                ## Query Your Documents
                
                Ask questions about the documents you've uploaded. The system will retrieve relevant chunks and generate answers with source citations.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="❓ Your Question",
                            lines=3,
                            placeholder="e.g., What are the main topics discussed?",
                            interactive=True
                        )
                    
                    with gr.Column(scale=1):
                        top_k_slider = gr.Slider(
                            label="📊 Number of Sources",
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            interactive=True
                        )
                        
                        model_dropdown = gr.Textbox(
                            label="🤖 Model (optional)",
                            value=DEFAULT_OLLAMA_MODEL,
                            placeholder="Leave blank for default",
                            interactive=True
                        )
                
                submit_btn = gr.Button(
                    "🔍 Search",
                    variant="primary",
                    scale=1
                )
                
                # Output
                answer_output = gr.Markdown(
                    label="📖 Answer with Sources",
                    value="Your answer will appear here... (Make sure to upload documents first!)"
                )
                
                # Connect button to function
                submit_btn.click(
                    fn=rag_demo_interface,
                    inputs=[query_input, top_k_slider, model_dropdown],
                    outputs=answer_output
                )
                
                # Example questions
                # gr.Examples(
                #     examples=[
                #         ["What are the main topics?", 5, DEFAULT_OLLAMA_MODEL],
                #         ["Summarize the key points", 5, DEFAULT_OLLAMA_MODEL],
                #         ["What's discussed in the document?", 5, DEFAULT_OLLAMA_MODEL],
                #         ["Give me a detailed overview", 5, DEFAULT_OLLAMA_MODEL],
                #     ],
                #     inputs=[query_input, top_k_slider, model_dropdown],
                #     outputs=answer_output,
                #     fn=rag_demo_interface,
                #     cache_examples=False,
                # )
                
                # Footer info
                gr.Markdown("""
                ---
                ### ℹ️ How It Works
                1. **Query Processing**: Your question is converted to embeddings
                2. **Semantic Search**: Top-K most relevant chunks are retrieved from documents
                3. **LLM Generation**: Local Ollama model generates answer based on retrieved chunks
                4. **Source Citation**: Answer includes [1], [2] citations linked to source documents
                
                ### 🚀 Performance
                - First run: ~5 minutes (document processing + embedding generation)
                - Subsequent queries: <5 seconds (embeddings cached)
                - Completely private & offline (when using local models)
                """)
    
    return demo


# # ===== 1. Configuration & Setup =====

# # This cell contains all the core logic: document loading, chunking, embedding, etc.
# # It must be run first to define all necessary functions.

# # (The code from your previous cells is assumed to be here)
# # ... DocumentChunk, MetadataStore, VectorStore, etc. ...

# # ===== DYNAMIC TOKEN LIMITS CONFIGURATION =====
# # Test the dynamic token limits with different numbers of sources

# print("=" * 70)
# print("🧪 DYNAMIC TOKEN LIMITS CONFIGURATION")
# print("=" * 70)

# # Test with different numbers of sources
# test_queries = [
#     ("What are the Four Laws of Behavior Change?", 3, "3 sources"),
#     ("How do habits compound over time?", 5, "5 sources"),
#     ("Explain the habit loop and its components", 8, "8 sources"),
#     ("What are the best strategies for building good habits?", 10, "10 sources (max)"),
# ]

# print("\n📊 Token Limit Configuration:")
# print(f"   Base tokens: {TOKEN_CONFIG['base']} (increased from 256)")
# print(f"   Per source: {TOKEN_CONFIG['per_source']} tokens/source")
# print(f"   Max tokens: {TOKEN_CONFIG['max']} (absolute cap)")
# print(f"\n📐 Formula: num_predict = min(base + num_sources × per_source, max)")

# print("\n" + "=" * 70)
# print("Expected token limits per test case:")
# print("=" * 70)

# for query, num_sources, description in test_queries:
#     expected_tokens = min(
#         TOKEN_CONFIG['base'] + (num_sources * TOKEN_CONFIG['per_source']),
#         TOKEN_CONFIG['max']
#     )
#     print(f"   {description:20} → {expected_tokens:4} max tokens")

# print("\n✅ All functions and configurations are now defined.")
# print("   Proceed to Step 2 to initialize the RAG system.")


# # ===== 3. Manual Test (Optional) =====

# print("\n" + "=" * 70)
# print("✅ QUICK TEST: Manual Query with 10 Sources")
# print("=" * 70)

# # Test directly with the loaded RAG system (no Gradio needed)
# if _demo_state.get("initialized"):
#     test_query = "What are the four laws of behavior change?"
#     print(f"\n📝 Query: {test_query}")
#     print(f"📊 Testing with: 10 sources (should show '[LLM Config] Sources: 10 | Max tokens: 2048')\n")
    
#     vs = _demo_state["vector_store"]
#     ms = _demo_state["metadata_store"]
#     result = execute_query(test_query, vs, ms, top_k=10)
    
#     print(f"\n✅ Answer length: {len(result['answer'])} characters")
#     print(f"✅ Sources retrieved: {len(result['retrieved_chunks'])}")
#     print("\n" + "=" * 70)
#     print("FORMATTED ANSWER WITH ALL SOURCES:")
#     print("=" * 70)
#     print(result["formatted_answer"][:1500])  # Show first 1500 chars
#     print("\n... (truncated for display)")
# else:
#     print("⚠️  RAG system not initialized. Run the 'Initialize RAG System' cell first.")
#     print("   If no saved data exists, you must first run the Gradio UI and upload documents.")


# # ===== 2. Initialize RAG System (Load Saved Data) =====

# print("=" * 70)
# print("Attempting to initialize RAG system from saved data...")
# print("=" * 70)

# try:
#     SAVE_DIR = DOCUMENT_CONFIG["save_dir"]
    
#     # Try to load previously saved data
#     vector_store = VectorStore.load(SAVE_DIR)
#     metadata_store = MetadataStore.load(os.path.join(SAVE_DIR, "metadata.pkl"))
    
#     # Update demo state for use in other cells
#     _demo_state["vector_store"] = vector_store
#     _demo_state["metadata_store"] = metadata_store
#     _demo_state["initialized"] = True
    
#     print(f"\n✅ RAG system initialized successfully from {SAVE_DIR}")
#     print(f"   - Loaded {vector_store.index.ntotal} vectors")
#     print(f"   - Loaded metadata for {len(metadata_store.chunks)} chunks")
#     print("\n✅ You can now run the 'Manual Test' cell or the 'Gradio UI' cell.")

# except FileNotFoundError:
#     print(f"\n⚠️  No saved data found in {DOCUMENT_CONFIG['save_dir']}")
#     print("\nACTION REQUIRED:")
#     print("1. Run the 'Start Gradio UI' cell below.")
#     print("2. Use the '📤 Upload Documents' tab to upload and process files.")
#     print("3. This will create the necessary data for other cells to use.")
# except Exception as e:
#     print(f"\n❌ Error loading vector store: {e}")
#     import traceback
#     traceback.print_exc()


# # ===== LAUNCH GRADIO DEMO =====

# print("=" * 70)
# print("🚀 Starting Gradio RAG Demo...")
# print("=" * 70)

# try:
#     # Close any existing demo to prevent conflicts
#     try:
#         if 'gradio_demo' in globals() and gradio_demo is not None:
#             print("⚠️  Closing previous Gradio instance...")
#             try:
#                 gradio_demo.close()
#             except Exception:
#                 pass  # Ignore errors when closing
#             # Wait a moment for cleanup
#             import time
#             time.sleep(1)
#     except Exception as cleanup_error:
#         print(f"  (Cleanup note: {cleanup_error})")
    
#     # Find available port (handle port conflicts gracefully)
#     preferred_port = 7860
#     if not is_port_available(preferred_port):
#         print(f"⚠️  Port {preferred_port} is already in use. Finding available port...")
#         available_port = find_available_port(preferred_port)
#         print(f"✓ Using port {available_port} instead\n")
#     else:
#         available_port = preferred_port
#         print(f"✓ Port {preferred_port} is available\n")
    
#     # Create and launch Gradio interface
#     gradio_demo = create_gradio_demo()
    
#     print("🌐 Launching Gradio interface...")
#     print("=" * 70)
#     print(f"📖 Access the demo at: http://127.0.0.1:{available_port}")
#     print("=" * 70)
#     print("\n📚 Instructions:")
#     print("1. Go to the '📤 Upload Documents' tab")
#     print("2. Upload your PDF, TXT, MD, or DOCX files")
#     print("3. Click '⚙️ Process Documents' and wait for completion")
#     print("4. Switch to '🔍 Ask Questions' tab to query your documents")
#     print("\n✨ Features:")
#     print("  • UTF-8 encoding fixed (proper quotes, dashes, special chars)")
#     print("  • Markdown rendering (code blocks, bold, bullets, etc.)")
#     print("  • Source citations with page numbers")
#     print("  • Hybrid search (local + web when enabled)")
#     print("\n✨ Your documents will be processed and indexed for semantic search!")
#     print("=" * 70 + "\n")
    
    
#     # Note: Skipping .queue() to avoid event loop conflicts in Jupyter
#     # Queue is not necessary for single-user notebook usage
    
#     # Launch with proper settings for Jupyter environment
#     gradio_demo.launch(
#         share=False,
#         server_name="0.0.0.0",
#         server_port=available_port,
#         prevent_thread_lock=True,
#         show_error=True,
#         quiet=False
#     )
    
# except Exception as e:
#     print(f"\n❌ Error launching Gradio demo: {e}")
#     print("\nTroubleshooting:")
#     print("  1. Event loop issue: Restart the Jupyter kernel completely")
#     print("  2. Port conflict: Check if port is already in use")
#     print("  3. Ollama not running: Run 'ollama serve' in another terminal")
#     print("  4. Try running this cell again")
#     import traceback
#     traceback.print_exc()


def initialize_rag_system():
    """
    Initializes the RAG system by loading data from disk.
    This is app-level logic, so it lives in app.py.
    """
    SAVE_DIR = DOCUMENT_CONFIG["save_dir"]
    try:
        if check_if_embeddings_exist(SAVE_DIR):
            print(f"✅ Loading existing vector store from {SAVE_DIR}...")
            vector_store = VectorStore.load(SAVE_DIR, verbose=False)
            metadata_store = MetadataStore.load(os.path.join(SAVE_DIR, "metadata.pkl"), verbose=False)
            
            _demo_state["vector_store"] = vector_store
            _demo_state["metadata_store"] = metadata_store
            _demo_state["initialized"] = True
            print(f"✅ RAG system initialized with {vector_store.index.ntotal} vectors.")
        else:
            print(f"⚠️  No saved data found in {SAVE_DIR}. Please upload documents via the UI.")
    except FileNotFoundError:
        print(f"⚠️  No saved data found in {SAVE_DIR}. Please upload documents via the UI.")
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting Gradio RAG Demo...")
    print("=" * 70)

    # 1. Initialize the system: Try to load existing embeddings from ./rag_data
    initialize_rag_system()

    # 2. Find an available port
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    if not is_port_available(preferred_port):
        print(f"⚠️  Port {preferred_port} is in use. Finding new port...")
        available_port = find_available_port(preferred_port)
    else:
        available_port = preferred_port
    print(f"✓ Port {available_port} is available.")

    # 3. Create and launch the Gradio UI
    gradio_demo = create_gradio_demo()
    
    print("🌐 Launching Gradio interface...")
    print(f"📖 Access the demo at: http://0.0.0.0:{available_port}")
    
    gradio_demo.launch(
        server_name="0.0.0.0",
        server_port=available_port,
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true"
    )