import os
import json
import requests
import re
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict,List 
from PyPDF2 import PdfReader

# Load .env
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Memory Location (ONLY for notes - brain.py handles chunks separately)
MEMORY_PATH = Path(os.path.dirname(__file__)) / "memory.json"

# =====================================================================
# SAFE CALCULATOR
# =====================================================================
def use_calculator(query: str) -> str:
    """
    Evaluates basic math expressions safely.
    """
    try:
        # Extract mathematical expression using regex
        # Match numbers, operators, parentheses, and decimals
        math_pattern = r'[\d\.\s\+\-\*\/\(\)]+'
        matches = re.findall(math_pattern, query)
        
        if not matches:
            return "No mathematical expression found. Example: 'Calculate 15 + 27' or 'What is 45 * 3?'"
        
        # Take the longest match (most likely the actual expression)
        expression = max(matches, key=len).strip()
        
        # Safety checks
        dangerous_patterns = [
            r'import\s', r'exec\s', r'eval\s', r'__', r'open\s', r'file\s',
            r'os\.', r'sys\.', r'subprocess', r'importlib'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return "Calculation contains unsafe patterns. Only basic arithmetic is allowed."
        
        # Only allow safe characters
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Only basic arithmetic operations (+, -, *, /, parentheses, decimals) are allowed."
        
        # Ensure the expression has at least one digit and one operator
        if not any(c.isdigit() for c in expression):
            return "No numbers found in the expression."
        
        if not any(op in expression for op in '+-*/'):
            return "No arithmetic operator found. Use +, -, *, or /"
        
        # Remove extra spaces
        expression = ''.join(expression.split())
        
        # Evaluate safely with limited globals
        result = eval(expression, {"__builtins__": {}}, {})
        
        return f"**Calculation Result:**\n\n`{expression} = {result}`"
    
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except SyntaxError:
        return "Invalid mathematical expression. Please check your syntax."
    except Exception as e:
        return f"Could not calculate: {str(e)}\n\nTry simpler expressions like '15 + 27' or '45 * 3'."

# =====================================================================
# NOTES HELPERS (Simplified - only for notes)
# =====================================================================
def load_notes() -> Dict:
    """Load only notes from memory.json."""
    if not MEMORY_PATH.exists():
        return {"notes": []}
    
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Ensure notes key exists
        if "notes" not in data:
            data["notes"] = []
        
        return data
    except json.JSONDecodeError:
        # Corrupted file, reset it
        print(f"[Tools] Corrupted memory file, resetting...")
        return {"notes": []}
    except Exception as e:
        print(f"[Tools] Error loading notes: {e}")
        return {"notes": []}

def save_note(text: str) -> str:
    """Save a new note."""
    if not text.strip():
        return "Note is empty. Nothing to save."
    
    data = load_notes()
    
    # Limit number of notes to prevent memory issues
    max_notes = 100
    if len(data["notes"]) >= max_notes:
        # Remove oldest note
        data["notes"].pop(0)
    
    data["notes"].append(text.strip())
    
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Return confirmation
        if len(text) > 50:
            return f"✅ Note saved: '{text[:50]}...'"
        else:
            return f"✅ Note saved: '{text}'"
    except Exception as e:
        return f"❌ Error saving note: {str(e)}"

def read_notes() -> List[str]:
    """Read all notes."""
    data = load_notes()
    return data.get("notes", [])

# =====================================================================
# WEB SEARCH (SERPER)
# =====================================================================
def search_web(query: str) -> str:
    """Search the web using Serper API."""
    if not SERPER_API_KEY:
        return "❌ SERPER_API_KEY missing in .env file. Please add it to use web search."
    
    if not query or not query.strip():
        return "Please provide a search query."
    
    query = query.strip()
    url = "https://google.serper.dev/search"
    
    payload = {"q": query}
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        results = response.json()
        
        # Check if we have organic results
        if "organic" not in results or not results["organic"]:
            # Check for answer box
            if "answerBox" in results:
                answer = results["answerBox"]
                title = answer.get("title", "")
                answer_text = answer.get("answer", "") or answer.get("snippet", "")
                return f"**{title}**\n\n{answer_text}"
            
            return f"No search results found for '{query}'."
        
        # Get top 5 results
        top_results = results["organic"][:5]
        
        # Format results
        output = f"🔍 **Search Results for '{query}':**\n\n"
        
        for i, item in enumerate(top_results, 1):
            title = item.get("title", "No Title")
            snippet = item.get("snippet", "No description available.")
            link = item.get("link", "")
            
            output += f"**{i}. {title}**\n"
            output += f"{snippet}\n"
            output += f"🔗 {link}\n\n"
        
        # Add answer box if available
        if "answerBox" in results:
            answer = results["answerBox"]
            answer_title = answer.get("title", "")
            answer_text = answer.get("answer", "") or answer.get("snippet", "")
            if answer_title or answer_text:
                output += "---\n"
                output += "**Quick Answer:**\n"
                if answer_title:
                    output += f"{answer_title}\n"
                if answer_text:
                    output += f"{answer_text}\n"
        
        return output
    
    except requests.exceptions.Timeout:
        return "⏰ Search timeout. Please try again in a moment."
    except requests.exceptions.ConnectionError:
        return "🔌 Connection error. Please check your internet connection."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "❌ Invalid SERPER_API_KEY. Please check your API key in .env file."
        return f"❌ HTTP Error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"❌ Search error: {str(e)}"

# =====================================================================
# PDF TEXT EXTRACTION (Improved)
# =====================================================================
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extracts text from PDF uploaded in Streamlit.
    Returns extracted text or error message string.
    """
    try:
        # Method 1: Try PyPDF2
        reader = PdfReader(uploaded_file)
        total_pages = len(reader.pages)
        
        if total_pages == 0:
            return "PDF has no pages."
        
        text = ""
        pages_with_text = 0
        
        for page_num in range(total_pages):
            try:
                page = reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text and page_text.strip():
                    text += f"--- Page {page_num + 1} ---\n{page_text.strip()}\n\n"
                    pages_with_text += 1
            except Exception as page_error:
                print(f"[Tools] Error reading page {page_num + 1}: {page_error}")
                text += f"--- Page {page_num + 1} ---\n[Error reading this page]\n\n"
        
        # Check if we got any text
        if not text.strip():
            return "PDF has no extractable text (may be scanned or image-based)."
        
        print(f"[Tools] Extracted text from {pages_with_text}/{total_pages} pages")
        
        # Method 2: Try PyMuPDF if available (for better extraction)
        try:
            import fitz  # PyMuPDF
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            fitz_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text and page_text.strip():
                    fitz_text += f"--- Page {page_num + 1} ---\n{page_text.strip()}\n\n"
            
            doc.close()
            
            # Use PyMuPDF text if it extracted more
            if len(fitz_text.strip()) > len(text.strip()):
                print(f"[Tools] PyMuPDF extracted more text ({len(fitz_text)} chars vs {len(text)} chars)")
                return fitz_text.strip()
            
        except ImportError:
            print("[Tools] PyMuPDF not installed, using PyPDF2 only")
        except Exception as fitz_error:
            print(f"[Tools] PyMuPDF error: {fitz_error}")
        
        return text.strip()
        
    except Exception as e:
        error_msg = f"Error reading PDF: {str(e)}"
        print(f"[Tools] {error_msg}")
        
        # Provide helpful suggestions
        if "cannot work with opened pdf" in str(e).lower() or "file has not been decrypted" in str(e).lower():
            error_msg += "\n\nThis PDF might be encrypted or password-protected."
        elif "pdf header signature not found" in str(e).lower():
            error_msg += "\n\nThis might not be a valid PDF file."
        
        return error_msg
    
    
    
    
    


# ====================================================================





