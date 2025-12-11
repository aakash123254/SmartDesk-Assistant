# test_rag.py
import json
from pathlib import Path
from agent import brain

MEM = Path(__file__).parent / "agent" / "memory.json"

def reset_memory():
    if MEM.exists():
        MEM.unlink()
    brain.save_memory({"notes": [], "chunks": []})

def add_sample_chunk():
    sample_text = (
        "This document describes the return policy. "
        "You can return items within 30 days with a receipt. "
        "Refunds will be processed to the original payment method."
    )
    chunks = brain.chunk_text(sample_text)
    mem = brain.load_memory()
    for c in chunks:
        emb = brain.embed_text(c)
        mem.setdefault("chunks", []).append({
            "filename": "sample_doc.txt",
            "text": c,
            "embedding": emb.tolist()
        })
    brain.save_memory(mem)
    print(f"Added {len(chunks)} sample chunks to memory.")

def test_rag_query():
    q = "What is the return window for items?"
    r = brain.rag_query(q)
    print("RAG result:", r)

def test_agent_fallback():
    q = "Say hello in a friendly way."
    out = brain.ask_agent(q)
    print("Agent output:", out[:500])

if __name__ == "__main__":
    reset_memory()
    add_sample_chunk()
    test_rag_query()
    test_agent_fallback()
