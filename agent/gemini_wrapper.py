import os
import time
import random
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file!")
    print("Please add: GEMINI_API_KEY=your_actual_key_here")
    # Don't raise error, just continue without Gemini
    gemini_available = False
else:
    try:
        genai.configure(api_key=api_key)
        gemini_available = True
        print("✅ Gemini API configured successfully")
    except Exception as e:
        print(f"❌ Gemini configuration error: {e}")
        gemini_available = False

# Use correct model name
MODEL_NAME = "gemini-2.0-flash"
model = None
if gemini_available:
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"✅ Gemini model '{MODEL_NAME}' loaded")
    except Exception as e:
        print(f"❌ Error loading Gemini model: {e}")
        gemini_available = False

# Fallback responses for when Gemini is unavailable
FALLBACK_RESPONSES = {
    "what is rag?": "RAG (Retrieval-Augmented Generation) is an AI framework that retrieves facts from an external knowledge base to improve the quality of generated responses. It combines the power of large language models with external information retrieval.",
    "what is mcp?": "MCP (Model Context Protocol) is a protocol for building AI applications that can interact with various tools and data sources through a standardized interface.",
    "hello": "Hello! I'm your assistant. Currently, I'm experiencing issues with my main AI service, but I can still help with basic tasks and answer simple questions.",
    "hi": "Hi there! I'm here to help. Note: My advanced AI features are temporarily unavailable.",
}

def get_fallback_response(query: str) -> str:
    """Get a fallback response when Gemini is unavailable."""
    query_lower = query.lower().strip()
    
    # SPECIAL CASE: If this looks like a RAG prompt (has "DOCUMENT EXCERPTS" or "ANSWER BASED ON DOCUMENTS")
    if "document excerpts" in query_lower or "answer based on documents" in query_lower:
        # Try to extract the query and context from the RAG prompt
        try:
            # Extract the user question (simplified parsing)
            if "USER QUESTION:" in query:
                parts = query.split("USER QUESTION:")
                if len(parts) > 1:
                    question_part = parts[1].split("IMPORTANT INSTRUCTIONS:")[0].strip()
                    context_part = query.split("DOCUMENT EXCERPTS:")[1].split("USER QUESTION:")[0].strip()
                    
                    # Return a basic answer based on context
                    return f"Based on the documents:\n\n{context_part[:1000]}...\n\n*Note: AI service is currently unavailable for detailed analysis.*"
        except:
            pass
    
    # Check for exact matches
    for key, response in FALLBACK_RESPONSES.items():
        if key in query_lower:
            return f"⚠️ (Gemini Unavailable) {response}"
    
    # Check for general topics
    if any(word in query_lower for word in ["rag", "retrieval", "augmented"]):
        return "⚠️ (Gemini Unavailable) RAG stands for Retrieval-Augmented Generation. It's a technique that enhances AI responses by retrieving relevant information from external sources before generating answers."
    
    if any(word in query_lower for word in ["mcp", "model context protocol"]):
        return "⚠️ (Gemini Unavailable) MCP (Model Context Protocol) is a standard protocol that allows AI models to interact with various tools, data sources, and applications through a consistent interface."
    
    if any(word in query_lower for word in ["machine learning", "ml", "ai", "artificial intelligence"]):
        return "⚠️ (Gemini Unavailable) Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
    
    # Generic fallback
    return f"⚠️ I'm currently experiencing technical difficulties with my AI service. Your question was: '{query}'. Please try again later or ask about specific topics, or use the calculator/notes features which are still working."

def _extract_text(response):
    """
    Safely extracts text from Gemini responses.
    """
    try:
        # Method 1: Direct text attribute
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        
        # Method 2: Extract from candidates
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            parts.append(part.text)
                    if parts:
                        return " ".join(parts).strip()
        
        # Method 3: Try to get any string representation
        text = str(response)
        if text and len(text.strip()) > 10:  # Avoid short error messages
            return text.strip()
        
        return "No response generated."
    
    except Exception as e:
        print(f"[Gemini] Error extracting text: {e}")
        return "Error extracting response."

def ask_gemini(prompt: str, max_retries: int = 1) -> str:
    """
    Sends a prompt to Gemini and returns the cleaned response.
    Includes retry logic for rate limits.
    """
    if not gemini_available or model is None:
        return get_fallback_response(prompt)
    
    if not prompt or not prompt.strip():
        return "Empty prompt provided."
    
    prompt = prompt.strip()
    
    for attempt in range(max_retries + 1):
        try:
            print(f"[Gemini] Sending prompt ({len(prompt)} chars)...")
            start_time = time.time()
            
            # Generate content with safety settings
            generation_config = {
                "temperature": 0.2,  # Lower for more consistent, factual responses
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,  # Reduced to save quota
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            elapsed = time.time() - start_time
            print(f"[Gemini] Response received in {elapsed:.2f}s")
            
            extracted_text = _extract_text(response)
            
            # Check for blocked responses
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                if hasattr(response.prompt_feedback, "block_reason"):
                    block_reason = response.prompt_feedback.block_reason
                    if block_reason:
                        return f"⚠️ Response blocked: {block_reason}. Using fallback response."
                        return get_fallback_response(prompt)
            
            # Check for empty or very short responses
            if not extracted_text or len(extracted_text.strip()) < 5:
                if attempt < max_retries:
                    print(f"[Gemini] Empty response, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)  # Brief pause before retry
                    continue
                return get_fallback_response(prompt)
            
            return extracted_text
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"[Gemini] Attempt {attempt + 1} failed: {error_msg}")
            
            # Rate limiting or quota errors
            if any(keyword in error_msg for keyword in ["quota", "rate limit", "429", "resource exhausted"]):
                print(f"[Gemini] Quota exceeded, using fallback responses")
                return get_fallback_response(prompt)
            
            # Authentication errors
            elif "api key" in error_msg or "401" in error_msg or "permission" in error_msg:
                print(f"[Gemini] API key error, using fallback")
                return get_fallback_response(prompt)
            
            # Model-specific errors
            elif "model" in error_msg and "not found" in error_msg:
                print(f"[Gemini] Model not found, using fallback")
                return get_fallback_response(prompt)
            
            # Network errors
            elif any(keyword in error_msg for keyword in ["timeout", "connection", "network"]):
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                print(f"[Gemini] Network error, using fallback")
                return get_fallback_response(prompt)
            
            # Other errors - return fallback
            print(f"[Gemini] Unknown error, using fallback")
            return get_fallback_response(prompt)
    
    return get_fallback_response(prompt)

def call_gemini(prompt: str) -> str:
    """
    Compatibility wrapper used inside brain.py.
    """
    return ask_gemini(prompt)

# Test function
def test_gemini():
    """Test the Gemini connection."""
    print("Testing Gemini connection...")
    try:
        response = ask_gemini("Hello, please respond with 'Gemini is working!'")
        if "Gemini is working" in response or "working" in response.lower():
            print("✅ Gemini connection successful!")
            return True
        else:
            print(f"⚠️ Using fallback: {response}")
            return False
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

if __name__ == "__main__":
    test_gemini()
    




