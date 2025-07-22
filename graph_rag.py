# Refactored ZepGraphRAG using functions instead of a class

import uuid
import asyncio
from typing import List, Dict, Any
from zep_python.client import AsyncZep
from zep_python.types import Message, SearchScope
import json
from pathlib import Path

# Globals for client, user_id, session_id
client: AsyncZep = None
user_id: str = None
session_id: str = None

def init_client(api_key: str, base_url: str = "http://localhost:8000") -> AsyncZep:
    global client
    client = AsyncZep(api_key=api_key, base_url=base_url)
    return client

async def setup_user_session(user_email: str = "user@example.com", user_name: str = "RAG User") -> tuple[str, str]:
    global user_id, session_id
    user_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex

    try:
        await client.user.add(
            user_id=user_id,
            email=user_email,
            first_name=user_name.split()[0],
            last_name=user_name.split()[-1] if " " in user_name else "",
            metadata={"purpose": "graph_rag"}
        )
    except Exception as e:
        print(f"User creation failed: {e}")

    try:
        await client.memory.add_session(
            session_id=session_id,
            user_id=user_id,
            metadata={"purpose": "document_rag"}
        )
    except Exception as e:
        print(f"Session creation failed: {e}")

    return user_id, session_id

def split_text_to_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        for i in range(end - 50, end):
            if i > start and text[i] in '.!?':
                end = i + 1
                break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks

async def ingest_document(file_path: str, chunk_size: int = 1000) -> int:
    if not session_id:
        raise ValueError("Session not set")
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    chunks = split_text_to_chunks(text, chunk_size)
    for i, chunk in enumerate(chunks):
        try:
            await client.memory.add_memory(
                session_id=session_id,
                messages=[
                    Message(
                        role_type="system",
                        role="DocumentChunk",
                        content=chunk,
                        metadata={"source_file": str(file_path), "chunk_id": i}
                    )
                ]
            )
        except Exception as e:
            print(f"Error on chunk {i}: {e}")
    await asyncio.sleep(2)
    return len(chunks)

def dict_to_natural_language(data: Dict[str, Any], data_type: str) -> str:
    sentences = []
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)):
            sentences.append(f"The {data_type} has {key} = {value}.")
        elif isinstance(value, list):
            sentences.append(f"The {data_type} includes {key}: {', '.join(map(str, value))}.")
        elif isinstance(value, dict):
            details = ", ".join(f"{k}: {v}" for k, v in value.items())
            sentences.append(f"The {data_type} has nested {key}: {details}.")
    return " ".join(sentences)

async def ingest_structured_data(data: List[Dict[str, Any]], data_type: str = "business_data") -> int:
    if not session_id:
        raise ValueError("Session not set")
    count = 0
    for i, record in enumerate(data):
        try:
            content = dict_to_natural_language(record, data_type)
            await client.memory.add_memory(
                session_id=session_id,
                messages=[
                    Message(
                        role_type="system",
                        role="DataRecord",
                        content=content,
                        metadata={"record_id": i}
                    )
                ]
            )
            count += 1
        except Exception as e:
            print(f"Error on record {i}: {e}")
    await asyncio.sleep(1)
    return count

async def retrieve_relevant_context(query: str, max_facts: int = 10, search_scope: str = "facts") -> Dict[str, Any]:
    await client.memory.add_memory(
        session_id=session_id,
        messages=[Message(role_type="user", role="QueryUser", content=query)]
    )
    await asyncio.sleep(0.5)
    memory = await client.memory.get(session_id=session_id)
    search_response = await client.memory.search_sessions(
        user_id=user_id,
        search_scope=search_scope,
        text=query,
        limit=max_facts
    )
    return {
        "query": query,
        "session_facts": memory.relevant_facts or [],
        "global_search_results": search_response.results or [],
        "recent_messages": memory.messages[-5:] if memory.messages else [],
        "total_facts_found": len(memory.relevant_facts or []) + len(search_response.results or [])
    }

def summarize_context(context: Dict[str, Any]) -> str:
    return f"Found {context['total_facts_found']} relevant facts"

def format_context_for_response(context: Dict[str, Any]) -> str:
    parts = []
    if context['session_facts']:
        parts.append("## Session Facts")
        for i, fact in enumerate(context['session_facts'], 1):
            parts.append(f"{i}. {fact}")
    if context['global_search_results']:
        parts.append("\n## Global Search Results")
        for i, result in enumerate(context['global_search_results'], 1):
            if hasattr(result, 'fact'):
                parts.append(f"{i}. {result.fact}")
            elif hasattr(result, 'message'):
                parts.append(f"{i}. {result.message.content}")
    return "\n".join(parts)

def create_llm_prompt(query: str, context: str) -> str:
    return f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"""

async def query(question: str, max_facts: int = 10) -> str:
    context = await retrieve_relevant_context(question, max_facts)
    summary = summarize_context(context)
    formatted = format_context_for_response(context)
    prompt = create_llm_prompt(question, formatted)
    return f"""🔍 Query: {question}\n📊 {summary}\n📚 Info:\n{formatted}\n🤖 Prompt:\n{prompt}"""

async def get_knowledge_graph_stats() -> Dict[str, Any]:
    memory = await client.memory.get(session_id=session_id)
    all_facts = await client.memory.search_sessions(
        user_id=user_id,
        search_scope="facts",
        text="",
        limit=1000
    )
    return {
        "session_messages": len(memory.messages),
        "session_facts": len(memory.relevant_facts),
        "total_searchable_facts": len(all_facts.results)
    }


# Example usage and demo
async def demo_zep_graph_rag():
    """Demonstrate Zep Graph RAG capabilities"""
    
    # Initialize (you'll need to set these values)
    API_KEY = "z_1dWlkIjoiNzY2M2MyMDItMDA4My00YWM5LTk0ODItZTAxYmY3ODljYzk4In0.4G3YDD0pjJ94x9Ja1gnSB9UHRSz3hxpujGYzPH6a_VE5M4ztKmF93sQkWseIWX8Vhzwv8xvMhGGe0pU48qBmqg"  # Set this to your actual API key
    BASE_URL = "http://localhost:8000"  # Change if using Zep Cloud
    
    # Create Graph RAG instance
    zep_rag = ZepGraphRAG(api_key=API_KEY, base_url=BASE_URL)
    
    try:
        # Setup user and session
        user_id, session_id = await zep_rag.setup_user_session(
            user_email="demo@example.com",
            user_name="Demo User"
        )
        
        print(f"✅ Setup complete - User: {user_id[:8]}..., Session: {session_id[:8]}...")
        
        # Example 1: Ingest a document
        # Uncomment and modify path as needed
        # await zep_rag.ingest_document("path/to/your/document.txt")
        
        # Example 2: Ingest structured data
        sample_data = [
            {
                "id": 1,
                "name": "Artificial Intelligence Overview",
                "description": "AI is the simulation of human intelligence in machines that are programmed to think and learn.",
                "category": "Technology",
                "date": "2024-01-15",
                "topics": ["machine learning", "neural networks", "deep learning"]
            },
            {
                "id": 2,
                "name": "Machine Learning Fundamentals",
                "description": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
                "category": "Technology",
                "date": "2024-01-16",
                "topics": ["supervised learning", "unsupervised learning", "reinforcement learning"]
            }
        ]
        
        await zep_rag.ingest_structured_data(sample_data, "knowledge_article")
        
        # Allow time for processing
        await asyncio.sleep(3)
        
        # Example queries
        queries = [
            "What is artificial intelligence?",
            "Tell me about machine learning types",
            "What topics are related to AI?"
        ]
        
        for query in queries:
            print(f"\n{'='*60}")
            response = await zep_rag.query(query)
            print(response)
        
        # Show knowledge graph statistics
        print(f"\n{'='*60}")
        print("📈 Knowledge Graph Statistics:")
        stats = await zep_rag.get_knowledge_graph_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")

# Run the demo
if __name__ == "__main__":
    # Note: You'll need to set up Zep first:
    # 1. Install Zep: pip install zep-python
    # 2. Run Zep locally or use Zep Cloud
    # 3. Set your API key
    
    print("Zep Graph RAG Demo")
    print("Note: Make sure Zep is running and you have valid credentials")
    
    # Uncomment to run the demo:
    # asyncio.run(demo_zep_graph_rag())