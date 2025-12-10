import os 
import json 
import requests 
from dotenv import load_dotenv
from pathlib import Path 
from PyPDF2 import PdfReader 

# Load environment variables from .env
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Path to memory file
MEMORY_PATH = Path(os.path.dirname(__file__))/ "memory.json"

# -----------------------------------------------------------------
# 1️⃣ Calculator Tool
# -----------------------------------------------------------------
def use_calculator(query: str):
    """
    Evaluates simple math expression safely 
    """
    
    try:
        # Only allow numbers and basic operators 
        safe_query = "".join(ch for ch in query if ch in "0123456789+-*/(). ")
        
        result = eval(safe_query)
        return f"Result: {result}"
    except Exception:
        return "Could not calculate that. Please use simple math expression."

# -----------------------------------------------------------------
# 2️⃣ Notes Tools (Write + Read Notes)
# -----------------------------------------------------------------
def load_memory():
    if not MEMORY_PATH.exists():
        return {"notes":[], "documents":[]}
    with open(MEMORY_PATH,"r",encoding="utf-8") as f:
        return json.load(f)


def save_memory(data):
    with open(MEMORY_PATH,"w",encoding="ustf-8") as f:
        json.dump(data,f,indent=4,ensure_ascii=False)

def save_note(text:str):
    memory = load_memory()
    memory["notes"].append(text)
    save_memory(memory)
    return f"Saved note: {text}"

def read_notes():
    memory = load_memory()
    return memory.get("notes",[])


# -----------------------------------------------------------------
# 3️⃣ Web Search Tool (Serper Free API)
# -----------------------------------------------------------------
def search_web(queryL:str):
    if not SERPER_API_KEY:
        return "❌ SERPER_API_KEY missing is .env"
    url = "https://google.serper.dev/search"
    
    payload = {"q":query}
    headers = {"X-API_KEY":SERPER_API_KEY}
    
    try: 
        response = requests.post(url,json=payload,headers=headers)
        results = response.json()
        
        if "organic" not in results:
            return "No search results found."
        
        top = results["organic"][:5] # top 5 results 
        
        output = "🔎 Top Search Results: \n\n"
        for i,item in enumerate(top,1):
            title = item.get("title","No title")
            snippet = item.get("snippet","No Description")
            link = item.get("link","")
            
            output += f"**{i}. {title}**\n{snippet}\n{link}\n\n"
        return output 
    except Exception as e:
        return f"Web search error: {str{(e)}}"
    
# -----------------------------------------------------------------
# 4️⃣ PDF Text Extraction Tool
# -----------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """
    Reads and extracts text from a PDf uploaded in Streamlit.
    """
    try:
        reader = PdfReader(uploaded_file)
        text=""
        
        for page in reader.pages:
            text+=page.extract_text() or ""
        return text.strip()
    
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    