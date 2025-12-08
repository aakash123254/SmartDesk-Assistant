import os 
from dotenv import load_dotenv
import streamlit as st 

# load environment variables from .env (GEMINI_API_KEY, SERPER_API_KEY)
load_dotenv()

# Import the agent function we will implement in brain.py 
# These functions will be created in the next step 
# - ask_agent(query) :  retuern agents textual answer 
# - ingest_pdf(file) : ingest a PDF file into memory/RAG 
# - list_notes() : return saved notes (string or list)

from agent.brain import ask_agent, ingest_pdf, list_notes 

st.set_page_config(page_title="SmartDesk Assistant",layout="wide")
st.title("SmartDesk Assistant 🧠")

# show which model/key will be used (for convenience)
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    st.sidebar.success("Gemini key loaded ✅")
else:
    st.sidebar.success("No GEMINI_API_KEY found in .env — add it before running")
    
st.sidebar.markdown("## Controls")
show_memory = st.sidebar.checkbox("Show saved notes",value=False)
model_info = st.sidebar.markdown("Model : **Gemini 2.0 Flash**")

# Main chat area 
st.write("### Chat with your agent")
query = st.text_input("Ask anything or give a command",key="query_input")

col1,col2 = st.columns([3,1])

with col1:
    if st.button("Send"):
        if not query:
            st.warning("Please type a question or command first.")
        else:
            with st.spinner("Agent thinking..."):
                try:
                    response = ask_agent(query)
                except Exception as e:
                    st.error(f"Agent error: {e}")
                    response = None 
            
            if response:
                st.markdown("**Agent reply**")
                st.write(response)
 
with col2:
    st.write("### Tools")
    uploaded_pdf = st.file_uploader("Upload PDF to add to memory",type=["pdf"]) 
    if uploaded_pdf:
        if st.button("Ingest PDF"):
            with st.spinner("Ingesting PDF into memory..."):
                try:
                    ingest_result = ingest_pdf(uploaded_pdf)
                    st.success(f"Ingested: {ingest_result}")
                except Exception as e:
                    st.error(f"PDF ingest failed: {e}")

# Optional: show saved notes/memory
if show_memory:
    st.write("### Saved Notes / Memory")
    try:
        notes = list_notes()
        if not notes:
            st.info("No saved notes found.")
        else:
            # if notes is a list, show each; else print full text 
            if isinstance(notes,list):
                for i,n in enumerate(notes,1):
                    st.write(f"{i}. {n}")
            else:
                st.write(notes)
    except Exception as e:
        st.error(f"Could not read notes: {e}")

st.write("---")
st.caption("Built with Gemini 2.0 Flash + Local memory. We'll implement the agent logic next.")
