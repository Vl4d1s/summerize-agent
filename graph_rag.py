import os
from zep_python import ZepClient
from zep_python.memory.search_type import SearchType
from zep_python.vector import Document, DocumentMetadata

# קונפיגורציה בסיסית
client = ZepClient(base_url="http://localhost:8000", api_key="z_1dWlkIjoiNzY2M2MyMDItMDA4My00YWM5LTk0ODItZTAxYmY3ODljYzk4In0.4G3YDD0pjJ94x9Ja1gnSB9UHRSz3hxpujGYzPH6a_VE5M4ztKmF93sQkWseIWX8Vhzwv8xvMhGGe0pU48qBmqg")
session_id = "static-session"  # סשן קבוע אחד בלבד

# הטענת המסמך הראשוני
def ingest_document(text: str):
    # פיצול לצ'אנקים
    chunks = split_into_chunks(text)
    
    documents = [
        Document(
            content=chunk,
            metadata=DocumentMetadata(source_name="initial_document.txt")
        )
        for chunk in chunks
    ]
    
    client.document.add(documents=documents, session_id=session_id)
    print("Document ingested successfully.")

# שאילתת מידע מהגרף
def query(query_text: str):
    search_results = client.memory.search_memory(
        session_id=session_id,
        text=query_text,
        search_type=SearchType.HYBRID,  # או SEMANTIC
        top_k=3
    )
    
    relevant_chunks = [msg.content for msg in search_results.messages]
    return relevant_chunks

# עוזר לפצל את הטקסט לצ’אנקים
def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    with open("src/data/events.txt", "r", encoding="utf-8") as f:
        text = f.read()
    ingest_document(text)

    results = query("What happened on April 2?")
    for chunk in results:
        print(chunk)
