import json
import os
import re 
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
from sentence_transformers import SentenceTransformer
from agent.tools import (
    use_calculator,
    save_note,
    read_notes,
    search_web,
    extract_text_from_pdf
)
from agent.gemini_wrapper import call_gemini

# ======================================
# GLOBAL MODEL (Load once - fixes slow startup)
# ======================================
_embedding_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model():
    """Lazy load embedding model to speed up startup."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[Brain] Loading embedding model: {_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(_MODEL_NAME)
        print("[Brain] Embedding model loaded successfully")
    return _embedding_model

# ======================================
# PATH & CONFIG
# ======================================
BASE_DIR = Path(os.path.dirname(__file__))
MEMORY_PATH = BASE_DIR / "memory.json"  # Only for notes
CHUNKS_PATH = BASE_DIR / "chunks.json"  # Separate file for PDF chunks
SIMILARITY_THRESHOLD = 0.4  # Lowered for better retrieval
MAX_CHUNKS_RETURN = 3

# ======================================
# MEMORY HELPERS (Optimized - separate notes and chunks)
# ======================================
def load_memory():
    """Load only notes from memory.json."""
    if not MEMORY_PATH.exists():
        return {"notes": []}

    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "notes" not in data:
            data["notes"] = []
        return data
    except Exception as e:
        print(f"[Brain] Error loading memory: {e}")
        return {"notes": []}

def save_memory(data):
    """Save only notes to memory.json."""
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Brain] Error saving memory: {e}")

def load_chunks():
    """Load chunks separately to avoid loading large embeddings unnecessarily."""
    if not CHUNKS_PATH.exists():
        return []
    
    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"[Brain] Loaded {len(chunks)} chunks from storage")
        return chunks
    except Exception as e:
        print(f"[Brain] Error loading chunks: {e}")
        return []

def save_chunks(chunks):
    """Save chunks to separate file."""
    try:
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"[Brain] Saved {len(chunks)} chunks to storage")
    except Exception as e:
        print(f"[Brain] Error saving chunks: {e}")

def calculate_chunk_id(text: str, filename: str) -> str:
    """Create unique ID for chunk to prevent duplicates."""
    content = f"{filename}:{text[:200]}"
    return hashlib.md5(content.encode()).hexdigest()

# ======================================
# EMBEDDING FUNCTIONS (Optimized with batching)
# ======================================
def embed_text(text: str) -> List[float]:
    """Embed single text."""
    model = get_embedding_model()
    return model.encode(text, convert_to_tensor=False).tolist()

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts at once (much faster)."""
    if not texts:
        return []
    
    model = get_embedding_model()
    print(f"[Brain] Batch embedding {len(texts)} texts...")
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False, batch_size=32)
    return embeddings.tolist()


# ======================================
# LOCAL ANSWER GENERATOR (When Gemini fails)
# ======================================
def generate_local_answer(query: str, context: str) -> str:
    """
    Generate a simple answer from context when Gemini is unavailable.
    FIXED: Handles incomplete/cut-off sentences from PDF chunks.
    """
    print("[Brain] Using local answer generator (Gemini unavailable)")
    
    query_lower = query.lower().strip()
    
    # FIX 1: Better sentence splitting that handles cut-off text
    sentences = []
    
    # First, clean up the context
    clean_context = context
    
    # Remove chunk metadata
    clean_context = re.sub(r'\[Document \d+, Relevance: [\d.]+\]', '', clean_context)
    clean_context = re.sub(r'--- Page \d+ ---', '', clean_context)
    clean_context = re.sub(r'Page \d+:', '', clean_context)
    
    # Fix hyphenated line breaks (common in PDFs)
    clean_context = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', clean_context)  # Fix "rela-\ntionship"
    
    # Normalize whitespace
    clean_context = re.sub(r'\s+', ' ', clean_context).strip()
    
    # Now split into sentences PROPERLY
    # Split by sentence endings, but be careful with abbreviations
    sentence_endings = re.finditer(r'[.!?]+', clean_context)
    
    last_end = 0
    for match in sentence_endings:
        end_pos = match.end()
        sentence = clean_context[last_end:end_pos].strip()
        
        # Skip very short fragments
        if len(sentence) > 20:
            sentences.append(sentence)
        last_end = end_pos
    
    # Don't forget the last part
    if last_end < len(clean_context):
        last_part = clean_context[last_end:].strip()
        if len(last_part) > 20:
            sentences.append(last_part + '.')  # Add period to incomplete
    
    # FIX 2: If we still have bad sentences, try paragraph-based approach
    if not sentences or all(len(s) < 30 for s in sentences):
        # Fallback to paragraph chunks
        paragraphs = [p.strip() for p in clean_context.split('\n\n') if p.strip()]
        
        # Take paragraphs that seem relevant
        for para in paragraphs:
            if len(para) > 50:  # Reasonable paragraph length
                sentences.append(para)
    
    # FIX 3: Better keyword matching for machine learning
    if "machine learning" in query_lower or "ml" in query_lower:
        keywords = ["machine learning", "ml", "learn", "algorithm", "model", "train", 
                   "predict", "data", "pattern", "statistical", "intelligence"]
    elif "ai" in query_lower or "artificial" in query_lower:
        keywords = ["artificial intelligence", "ai", "intelligent", "system", "cognitive"]
    else:
        keywords = [word for word in query_lower.split() if len(word) > 3]
    
    # Find relevant sentences
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Score based on keyword matches
        score = 0
        for keyword in keywords:
            if keyword in sentence_lower:
                score += 1
        
        # Bonus points for definition patterns
        definition_patterns = ["is a", "are", "means", "refers to", "defined as", 
                              "involves", "entails", "consists of"]
        for pattern in definition_patterns:
            if pattern in sentence_lower:
                score += 2
                break
        
        if score > 0:
            # Clean the sentence
            cleaned = re.sub(r'\s+', ' ', sentence).strip()
            
            # Fix common PDF issues
            cleaned = re.sub(r'\b(\w+)-\s+(\w+)\b', r'\1\2', cleaned)  # Fix "ma- chine"
            
            if len(cleaned) > 25:  # Minimum length
                relevant_sentences.append((cleaned, score))
    
    # Sort by score (highest first)
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    
    if relevant_sentences:
        # Take top 3-4 sentences
        top_sentences = [s for s, _ in relevant_sentences[:4]]
        
        # Format answer
        answer = "**Based on your documents:**\n\n"
        
        # Group related sentences
        grouped = []
        current_group = ""
        
        for sentence in top_sentences:
            # If current group is short, append to it
            if len(current_group) + len(sentence) < 300:
                if current_group:
                    current_group += " " + sentence
                else:
                    current_group = sentence
            else:
                grouped.append(current_group)
                current_group = sentence
        
        if current_group:
            grouped.append(current_group)
        
        # Output as bullet points
        for i, group in enumerate(grouped, 1):
            answer += f"{i}. {group}\n\n"
        
        answer += "---\n*Note: AI service is currently unavailable for detailed analysis.*"
        return answer
    
    # FIX 4: Better fallback - extract key phrases
    words = clean_context.split()
    
    # Find 2-4 word phrases that might be meaningful
    phrases = []
    for i in range(len(words) - 3):
        phrase = ' '.join(words[i:i+4])
        phrase_lower = phrase.lower()
        
        # Check if phrase contains relevant terms
        if any(term in phrase_lower for term in ["machine", "learn", "algorithm", "model", "data"]):
            phrases.append(phrase)
    
    if phrases:
        unique_phrases = []
        seen = set()
        for phrase in phrases[:8]:  # Take first 8
            if phrase not in seen:
                seen.add(phrase)
                unique_phrases.append(phrase)
        
        answer = "**Key phrases from your documents:**\n\n"
        for phrase in unique_phrases:
            answer += f"• {phrase}\n"
        
        answer += "\n---\n*Note: Content extracted from PDF. AI service unavailable.*"
        return answer
    
    # Ultimate fallback
    if len(clean_context) > 1000:
        # Show beginning and end
        preview = clean_context[:400] + "\n\n[...]\n\n" + clean_context[-400:]
    else:
        preview = clean_context
    
    return f"**Document content:**\n\n{preview}\n\n*Unable to extract specific answer (AI service down)*"
# ======================================
# PDF CHUNKER (Improved - preserves context)
# ======================================
def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """Better chunking that preserves complete sentences and paragraphs."""
    # Clean text first
    text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
    text = re.sub(r'\s{2,}', ' ', text)     # Remove excessive spaces
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Skip very short paragraphs (page numbers, headers)
        if len(para) < 30:
            continue
            
        # If paragraph is complete and can stand alone
        if para.endswith('.') or para.endswith('?') or para.endswith('!'):
            is_complete = True
        else:
            # Check if it's likely a complete paragraph
            sentences = re.split(r'[.!?]+', para)
            is_complete = len(sentences) > 1 and len(para) > 50
            
        if is_complete:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
        else:
            # Incomplete paragraph, append to current
            if len(current_chunk) + len(para) + 1 <= max_chars:
                if current_chunk:
                    current_chunk += " " + para
                else:
                    current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Now ensure each chunk ends with complete sentences
    final_chunks = []
    for chunk in chunks:
        # Find the last sentence boundary
        last_period = chunk.rfind('.')
        last_question = chunk.rfind('?')
        last_exclamation = chunk.rfind('!')
        
        last_boundary = max(last_period, last_question, last_exclamation)
        
        if last_boundary > len(chunk) * 0.7:  # If we have a good boundary in last 30%
            # Cut at the boundary
            final_chunks.append(chunk[:last_boundary + 1].strip())
        else:
            # No good boundary, keep as is
            final_chunks.append(chunk)
    
    # Add overlap between chunks
    if len(final_chunks) > 1:
        overlapped_chunks = [final_chunks[0]]
        for i in range(1, len(final_chunks)):
            # Take last 100-150 chars from previous chunk
            prev_chunk = final_chunks[i-1]
            if len(prev_chunk) > overlap:
                # Try to find a sentence boundary
                overlap_start = max(0, len(prev_chunk) - overlap - 50)
                overlap_text = prev_chunk[overlap_start:]
                
                # Find first sentence boundary in overlap text
                first_period = overlap_text.find('.')
                if first_period > 20:  # Reasonable position
                    overlap_text = overlap_text[first_period + 1:].strip()
            else:
                overlap_text = prev_chunk
            
            combined = overlap_text + "\n\n" + final_chunks[i]
            overlapped_chunks.append(combined)
        return overlapped_chunks
    
    return final_chunks

# ======================================
# RAG QUERY LOGIC (Fixed - actually retrieves from PDF)
# ======================================
def rag_query(query: str, top_k: int = MAX_CHUNKS_RETURN) -> Dict[str, Any]:
    """
    Improved RAG query with better retrieval.
    Returns the best matching PDF content or None.
    """
    print(f"[Brain] RAG query: '{query[:50]}...'")
    
    chunks = load_chunks()
    
    if not chunks:
        print("[Brain] No chunks available for RAG")
        return {"source": "model", "answer": None, "score": 0, "chunks_used": 0}
    
    print(f"[Brain] Searching through {len(chunks)} chunks...")
    
    # Embed the query
    query_start = time.time()
    query_embedding = embed_text(query)
    query_embedding_np = np.array(query_embedding)
    query_time = time.time() - query_start
    print(f"[Brain] Query embedded in {query_time:.2f}s")
    
    best_matches = []
    
    # Search through chunks
    search_start = time.time()
    for chunk in chunks:
        try:
            chunk_embedding = chunk.get("embedding")
            if not chunk_embedding:
                continue
                
            chunk_embedding_np = np.array(chunk_embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding_np, chunk_embedding_np) / (
                np.linalg.norm(query_embedding_np) * np.linalg.norm(chunk_embedding_np) + 1e-10
            )
            
            if similarity > SIMILARITY_THRESHOLD:
                best_matches.append({
                    "text": chunk["text"],
                    "score": float(similarity),
                    "filename": chunk.get("filename", "unknown"),
                    "chunk_id": chunk.get("chunk_id", "unknown")
                })
                
        except Exception as e:
            print(f"[Brain] Error processing chunk: {e}")
            continue
    
    search_time = time.time() - search_start
    print(f"[Brain] Searched {len(chunks)} chunks in {search_time:.2f}s, found {len(best_matches)} matches")
    
    # Sort by score (highest first) and take top_k
    best_matches.sort(key=lambda x: x["score"], reverse=True)
    best_matches = best_matches[:top_k]
    
    if best_matches:
        print(f"[Brain] Best match score: {best_matches[0]['score']:.3f}")
        
        # Combine top matches
        context_parts = []
        for i, match in enumerate(best_matches):
            context_parts.append(f"[Document {i+1}, Relevance: {match['score']:.2f}]\n{match['text']}")
        
        combined_text = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided document excerpts.

DOCUMENT EXCERPTS:
{combined_text}

USER QUESTION:
{query}

IMPORTANT INSTRUCTIONS:
1. Answer using ONLY the information from the document excerpts above.
2. If the document doesn't contain information to answer the question, say: "The provided documents don't contain information about this."
3. Do NOT use any external knowledge or make assumptions.
4. Be precise and reference the document information.
5. Keep your answer concise and directly relevant to the question.

ANSWER BASED ON DOCUMENTS:"""
        
        try:
            answer_start = time.time()
            pdf_answer = call_gemini(prompt)
            answer_time = time.time() - answer_start
            
            # Check if Gemini returned a fallback/error response
            if "unavailable" in pdf_answer.lower() or "⚠️" in pdf_answer or "quota" in pdf_answer.lower():
                print("[Brain] Gemini unavailable, using local answer generator")
                # Use local generator with the combined text
                pdf_answer = generate_local_answer(query, combined_text)
            
            # Check if the answer is actually from documents
            if "doesn't contain information" in pdf_answer.lower():
                print("[Brain] Documents don't contain relevant info")
                return {"source": "model", "answer": None, "score": best_matches[0]["score"], "chunks_used": len(best_matches)}
            
            print(f"[Brain] Generated answer in {answer_time:.2f}s")
            return {
                "source": "pdf",
                "answer": pdf_answer,
                "score": best_matches[0]["score"],
                "chunks_used": len(best_matches),
                "filenames": list(set(match["filename"] for match in best_matches))
            }
        except Exception as e:
            print(f"[Brain] Error generating answer: {e}")
            # Try local generator as last resort
            try:
                pdf_answer = generate_local_answer(query, combined_text)
                return {
                    "source": "pdf",
                    "answer": pdf_answer,
                    "score": best_matches[0]["score"],
                    "chunks_used": len(best_matches),
                    "filenames": list(set(match["filename"] for match in best_matches))
                }
            except Exception as e2:
                print(f"[Brain] Local generator also failed: {e2}")
                return {"source": "model", "answer": None, "score": 0, "chunks_used": 0}

# ======================================
# MAIN AGENT ROUTER (FIXED - Better calculator detection)
# ======================================
def ask_agent(query: str) -> str:
    """Main agent function with improved tool detection."""
    print(f"[Brain] Agent query: '{query[:50]}...'")
    
    query_lower = query.lower().strip()
    
    # ---- Tool Detection ----
    # Calculator detection - ONLY trigger for actual math problems
    # Check if it's REALLY a math query (has numbers AND operators)
    has_numbers = any(c.isdigit() for c in query)
    has_math_operators = any(op in query for op in ['+', '-', '*', '/', '×', '÷', '='])
    math_keywords = ["calculate", "compute", "solve", "add", "subtract", "multiply", "divide", "plus", "minus", "times"]
    
    # Only use calculator if it's clearly a math problem
    is_math_query = (
        (has_numbers and has_math_operators) or  # Has both numbers and operators
        (any(word in query_lower for word in ["what is", "how much is"]) and has_numbers and has_math_operators) or  # "What is 5 + 3?"
        (any(word in query_lower for word in math_keywords) and has_numbers)  # "Calculate 15 times 3"
    )
    
    # Special case: "what is rag?" should NOT trigger calculator
    if "what is" in query_lower and not has_numbers:
        is_math_query = False
    
    if is_math_query:
        print("[Brain] Using calculator")
        return use_calculator(query)
    
    # Notes detection
    if query_lower.startswith(("add note", "save note", "remember that", "note:")):
        print("[Brain] Saving note")
        note = query.split("note", 1)[-1].strip() if "note" in query_lower else query
        return save_note(note)
    
    if query_lower in ["show notes", "list notes", "my notes", "what notes", "get notes"]:
        print("[Brain] Listing notes")
        notes = read_notes()
        if notes:
            return "📝 **Your Notes:**\n\n" + "\n".join([f"{i+1}. {note}" for i, note in enumerate(notes)])
        return "No saved notes yet."
    
    # Web search detection
    if query_lower.startswith(("search for", "google", "find info about", "look up", "search:")):
        print("[Brain] Web search")
        search_term = query_lower.replace("search for", "").replace("google", "").replace("find info about", "").replace("look up", "").replace("search:", "").strip()
        return search_web(search_term if search_term else query)
    
    # ---- Try RAG first ----
    print("[Brain] Trying RAG retrieval...")
    rag_result = rag_query(query)
    
    if rag_result["source"] == "pdf" and rag_result["answer"]:
        print(f"[Brain] Using PDF answer (score: {rag_result['score']:.3f})")
        # Check if the answer actually contains useful info
        if "doesn't contain information" not in rag_result["answer"].lower():
            return rag_result["answer"]
    
    # ---- Fallback to Gemini ----
    print("[Brain] Using Gemini fallback")
    model_answer = call_gemini(query)
    return model_answer

# ======================================
# PDF INGESTION (FIXED - Variable name conflict resolved)
# ======================================
def ingest_pdf(file_path: str) -> str:
    """Optimized PDF ingestion with batch embedding."""
    print(f"[Brain] Starting PDF ingestion: {file_path}")
    start_time = time.time()
    
    # Extract text - PASS FILE PATH DIRECTLY
    extract_start = time.time()
    
    try:
        # Open file in binary mode
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Create proper file-like object with seek() method
        class FileLikeObject:
            def __init__(self, content, name):
                self.content = content
                self.name = name
                self.position = 0
            
            def read(self, size=-1):
                if size == -1:
                    result = self.content[self.position:]
                    self.position = len(self.content)
                else:
                    result = self.content[self.position:self.position + size]
                    self.position += size
                return result
            
            def seek(self, position, whence=0):
                if whence == 0:
                    self.position = position
                elif whence == 1:
                    self.position += position
                elif whence == 2:
                    self.position = len(self.content) + position
                
                # Ensure position is within bounds
                self.position = max(0, min(self.position, len(self.content)))
                return self.position
            
            def tell(self):
                return self.position
            
            def getvalue(self):
                return self.content
        
        file_obj = FileLikeObject(file_content, os.path.basename(file_path))
        
        # Extract text
        pdf_text = extract_text_from_pdf(file_obj)
        
    except Exception as e:
        return f"❌ Error reading PDF file: {str(e)}"
    
    extract_time = time.time() - extract_start
    
    # Check if extraction failed
    if not pdf_text:
        return "❌ Failed to extract text from PDF (empty result)."
    
    if isinstance(pdf_text, str) and pdf_text.startswith("Error"):
        return f"❌ {pdf_text}"
    
    if "PDF appears to be scanned" in pdf_text:
        return f"⚠️ {pdf_text}"
    
    if not pdf_text.strip():
        return "❌ PDF is empty or has no extractable text."
    
    print(f"[Brain] Extracted {len(pdf_text)} characters in {extract_time:.2f}s")
    
    # Chunk the text
    chunk_start = time.time()
    text_chunks = chunk_text(pdf_text)  # Changed variable name from 'chunks' to 'text_chunks'
    chunk_time = time.time() - chunk_start
    
    print(f"[Brain] Created {len(text_chunks)} chunks in {chunk_time:.2f}s")
    
    if not text_chunks:
        return "❌ No valid chunks created from PDF text."
    
    # Load existing chunks
    existing_chunks = load_chunks()
    existing_ids = {chunk.get("chunk_id") for chunk in existing_chunks if "chunk_id" in chunk}
    
    # Prepare new chunks with unique IDs - FIXED VARIABLE NAME
    new_chunks_data = []
    for chunk_content in text_chunks:  # Changed variable name
        chunk_id = calculate_chunk_id(chunk_content, os.path.basename(file_path))
        
        # Skip duplicates
        if chunk_id in existing_ids:
            continue
            
        new_chunks_data.append({
            "text": chunk_content,  # Changed variable name
            "chunk_id": chunk_id,
            "filename": os.path.basename(file_path),
            "timestamp": time.time(),
            "text_length": len(chunk_content)  # Changed variable name
        })
    
    if not new_chunks_data:
        return f"✅ No new content to add from {os.path.basename(file_path)} (all chunks already exist)"
    
    print(f"[Brain] Prepared {len(new_chunks_data)} new chunks for embedding")
    
    # Batch embed all new chunks at once (MUCH faster)
    embed_start = time.time()
    chunk_texts = [chunk["text"] for chunk in new_chunks_data]
    embeddings = embed_batch(chunk_texts)
    embed_time = time.time() - embed_start
    
    print(f"[Brain] Embedded {len(new_chunks_data)} chunks in {embed_time:.2f}s")
    
    # Add embeddings to chunks
    for i, chunk in enumerate(new_chunks_data):
        chunk["embedding"] = embeddings[i]
    
    # Add to existing chunks
    existing_chunks.extend(new_chunks_data)
    
    # Save chunks
    save_start = time.time()
    save_chunks(existing_chunks)
    save_time = time.time() - save_start
    
    total_time = time.time() - start_time
    
    print(f"[Brain] Total ingestion time: {total_time:.2f}s")
    
    return f"✅ Successfully ingested {len(new_chunks_data)} chunks from {os.path.basename(file_path)} in {total_time:.1f}s. Total chunks: {len(existing_chunks)}"

# ======================================
# UTILITY FUNCTIONS
# ======================================
def list_notes() -> List[str]:
    """List all notes."""
    memory = load_memory()
    return memory.get("notes", [])

def list_chunks() -> List[Dict]:
    """List all chunks (without embeddings for display)."""
    chunks = load_chunks()
    # Remove embeddings for display to save space
    display_chunks = []
    for chunk in chunks[:20]:  # Limit to first 20 for display
        display_chunk = chunk.copy()
        if "embedding" in display_chunk:
            display_chunk.pop("embedding")
        display_chunks.append(display_chunk)
    return display_chunks

def clear_chunks() -> str:
    """Clear all PDF chunks."""
    save_chunks([])
    return "✅ All PDF chunks cleared."

def get_stats() -> Dict[str, Any]:
    """Get statistics about the knowledge base."""
    chunks = load_chunks()
    memory = load_memory()
    
    if chunks:
        text_lengths = [len(chunk.get('text', '')) for chunk in chunks]
        avg_chunk_size = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    else:
        avg_chunk_size = 0
        text_lengths = []
    
    return {
        "total_chunks": len(chunks),
        "total_notes": len(memory.get("notes", [])),
        "unique_files": len(set(chunk.get("filename", "unknown") for chunk in chunks)),
        "chunk_size_range": f"{min(text_lengths) if text_lengths else 0} - {max(text_lengths) if text_lengths else 0} chars",
        "avg_chunk_size": f"{avg_chunk_size:.0f} chars",
        "similarity_threshold": SIMILARITY_THRESHOLD
    }
    
    
    



















