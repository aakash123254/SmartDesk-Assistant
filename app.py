import os
import tempfile
import uuid
from dotenv import load_dotenv
import streamlit as st


# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "notes" not in st.session_state:
    st.session_state.notes = []

# ---------------------------------------
# LAZY LOAD BRAIN (Optimized)
# ---------------------------------------
@st.cache_resource
def load_brain():
    """Lazy load brain module to speed up startup."""
    try:
        from agent.brain import ask_agent, ingest_pdf, list_notes, rag_query, get_stats, clear_chunks, list_chunks
        return {
            "ask_agent": ask_agent,
            "ingest_pdf": ingest_pdf,
            "list_notes": list_notes,
            "rag_query": rag_query,
            "get_stats": get_stats,
            "clear_chunks": clear_chunks,
            "list_chunks": list_chunks
        }
    except ImportError as e:
        st.error(f"Failed to load brain module: {e}")
        return None

# Load brain functions
brain = load_brain()
if brain:
    ask_agent = brain["ask_agent"]
    ingest_pdf = brain["ingest_pdf"]
    list_notes = brain["list_notes"]
    rag_query = brain["rag_query"]
    get_stats = brain["get_stats"]
    clear_chunks = brain["clear_chunks"]
    list_chunks = brain["list_chunks"]

# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.set_page_config(page_title="SmartDesk Assistant", layout="wide")
st.title("SmartDesk Assistant 🧠")

# Sidebar
st.sidebar.title("Configuration")
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    st.sidebar.success("Gemini key loaded ✅")
else:
    st.sidebar.error("No GEMINI_API_KEY in .env file")
    st.sidebar.info("Add GEMINI_API_KEY to your .env file")

# Settings
st.sidebar.markdown("## Settings")
use_rag = st.sidebar.checkbox("Use RAG (PDF Search)", value=True, help="Search your uploaded PDFs for answers")
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

# Clear buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col2:
    if brain and st.button("Clear PDFs", use_container_width=True):
        msg = clear_chunks()
        st.sidebar.success(msg)
        st.rerun()
if brain:
    if st.sidebar.button("🗑️ Clear All PDF Data", type="secondary", use_container_width=True):
        try:
            from agent.brain import clear_chunks
            msg = clear_chunks()
            st.sidebar.success(msg)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to clear: {e}")

# Debug Info
if show_debug and brain:
    st.sidebar.markdown("---")
    st.sidebar.write("### RAG Stats")
    try:
        stats = get_stats()
        st.sidebar.metric("Total Chunks", stats['total_chunks'])
        st.sidebar.metric("Total Notes", stats['total_notes'])
        st.sidebar.metric("Unique Files", stats['unique_files'])
        st.sidebar.caption(f"Chunk Size: {stats['chunk_size_range']}")
        
        if stats['total_chunks'] > 0:
            if st.sidebar.button("View Chunks"):
                chunks = list_chunks()
                for i, chunk in enumerate(chunks[:5], 1):
                    st.sidebar.text_area(f"Chunk {i} ({chunk.get('filename', 'unknown')})", 
                                       chunk.get('text', '')[:200] + "...", 
                                       height=100)
    except Exception as e:
        st.sidebar.error(f"Stats error: {e}")

st.sidebar.markdown("---")
st.sidebar.write("### Model Info")
st.sidebar.markdown("- **Model**: Gemini 2.0 Flash")
st.sidebar.markdown("- **Embedding**: all-MiniLM-L6-v2")
st.sidebar.markdown(f"- **RAG**: {'✅ Enabled' if use_rag else '❌ Disabled'}")

# PDF Upload Section
st.sidebar.markdown("---")
st.sidebar.write("### PDF Upload")
uploaded_pdf = st.sidebar.file_uploader(
    "Choose PDF file", 
    type=["pdf"],
    help="Upload PDFs to enhance the agent's knowledge",
    label_visibility="collapsed"
)

if uploaded_pdf and brain:
    if st.sidebar.button("Ingest PDF", type="primary", use_container_width=True):
        # Save uploaded file to a temporary file
        import tempfile
        import uuid
        
        # Create unique filename
        temp_dir = tempfile.gettempdir()
        temp_filename = f"rag_pdf_{uuid.uuid4().hex}.pdf"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            # Save uploaded file to temp location
            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.getvalue())
            
            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                # Pass the file path to ingest_pdf
                msg = ingest_pdf(temp_path)
                
                if msg.startswith("✅"):
                    st.sidebar.success(msg)
                elif msg.startswith("⚠️"):
                    st.sidebar.warning(msg)
                else:
                    st.sidebar.error(msg)
                    
                # Update notes display
                try:
                    st.session_state.notes = list_notes()
                except:
                    pass
                    
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Display chat history
st.write("### Chat History")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            if message["source"] == "pdf":
                st.success(f"📄 Source: PDF (score: {message.get('score', 0):.2f})")
            elif message["source"] == "model":
                st.info("🤖 Source: Gemini")
            else:
                st.caption(f"Source: {message['source']}")

# Chat input
if prompt := st.chat_input("Ask your agent anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if not brain:
                    response = "Brain module not loaded. Check server logs."
                    source = "Error"
                    score = 0
                elif use_rag:
                    # Try RAG first
                    rag_result = rag_query(prompt)
                    
                    if rag_result["source"] == "pdf" and rag_result["answer"]:
                        response = rag_result["answer"]
                        source = "pdf"
                        score = rag_result.get("score", 0)
                    else:
                        # Fallback to agent
                        response = ask_agent(prompt)
                        source = "model"
                        score = 0
                else:
                    # Direct agent call
                    response = ask_agent(prompt)
                    source = "model"
                    score = 0
                
                # Display response
                st.markdown(response)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source": source,
                    "score": score
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "source": "Error"
                })

# Notes Section in main area
st.sidebar.markdown("---")
show_memory = st.sidebar.checkbox("Show Saved Notes", value=False)

if show_memory:
    st.write("### 📝 Saved Notes")
    
    if not st.session_state.notes:
        try:
            if brain:
                st.session_state.notes = list_notes()
            else:
                st.session_state.notes = []
        except:
            st.session_state.notes = []
    
    if st.session_state.notes:
        for i, note in enumerate(st.session_state.notes, 1):
            with st.expander(f"Note {i}"):
                st.write(note)
                
        # Add new note
        with st.form("add_note_form"):
            new_note = st.text_area("Add a new note:")
            if st.form_submit_button("Save Note"):
                if new_note.strip():
                    if brain:
                        result = ask_agent(f"save note: {new_note}")
                        st.success("Note saved!")
                        st.session_state.notes = list_notes()
                        st.rerun()
                    else:
                        st.error("Brain module not loaded")
    else:
        st.info("No notes saved yet. Try saying 'save note: [your note]'")

# Footer
st.markdown("---")
st.caption("Built with RAG + Gemini + Local Memory | SmartDesk Assistant v2.0 (Fixed)")



