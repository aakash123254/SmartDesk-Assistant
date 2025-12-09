import json 
import os 
from pathlib import Path 
from agent.tools import use_calculator,save_note,read_notes,search_web,extract_text_from_pdf 
from agent.gemini_wrapper import call_gemini 

# Path to memory file 
MEMORY_PATH = Path(os.path.dirname(__file__)) / "memory.json"

# ----------------------------
# Helper: load memory.json
# ----------------------------
def load_memory():
    if not MEMORY_PATH.exist():
        return {"notes" : [], "documents":[]}
    with open(MEMORY_PATH,"r",encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Helper: save memory.json
# ----------------------------
def save_memory(data):
    with open(MEMORY_PATH, "w",encoding="utf-8") as f:
        json.dump(data,f,indent=4,ensure_ascii=False)


# -------------------------------------------------------
# 1️⃣ MAIN AGENT FUNCTION — handles commands & LLM output
# -------------------------------------------------------
def ask_agent(query: str):
    query_lower = query.lower()
    
    # ----------------------------
    # TOOL ROUTING (simple rules)
    # ----------------------------
    
    # Calculator
    if any(op in query_lower for op in ["calculate","+","-","*","/","solve"]):
        return use_calculator(query)
    
    #Adding a note 
    if query_lower.startswith("add note") or query_lower.startswith("save note"):
        note = query.split("note",1)[-1].strip()
        return save_note(note)
    
    # Show notes 
    if "show notes" in query_lower or "list notes" in query_lower:
        notes = read_notes()
        if not notes:
            return "No saved notes yet."
        return "\n".join(f"- {n}" for n in notes)
    
    # Web search 
    if query_lower.startswith("search") or "google" in query_lower:
        search_query = query.replace("search","").replace("google","").strip()
        return search_web(search_query)
    
    # if no special tool is triggered → use Gemini LLM 
    response = call_gemini(query)
    return response

# -------------------------------------------------------
# 2️⃣ PDF INGEST FUNCTION (RAG Storage)
# -------------------------------------------------------
def ingest_pdf(uploaded_file):
    memory = load_memory()
    
    #Extract text 
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Add to memory under "documents"
    memory["documents"].append({
        "filename" : uploaded_file.name,
        "content" : pdf_text
    })
    
    # Save memory 
    save_memory(memory)
    
    return f"Stored {uploaded_file.name} with {len(pdf_text)} characters." 

# -------------------------------------------------------
# 3️⃣ List notes (for Streamlit sidebar)
# -------------------------------------------------------
def list_notes():
    memory = load_memory()
    return memory.get("notes",[])
