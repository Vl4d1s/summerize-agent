import os
from zep_cloud.client import Zep
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from src.prompts import create_qna_prompt

# Basic configuration
client = Zep(api_key="z_1dWlkIjoiNzY2M2MyMDItMDA4My00YWM5LTk0ODItZTAxYmY3ODljYzk4In0.4G3YDD0pjJ94x9Ja1gnSB9UHRSz3hxpujGYzPH6a_VE5M4ztKmF93sQkWseIWX8Vhzwv8xvMhGGe0pU48qBmqg")  # Replace with your actual API key
user_id = "static-user"
session_id = "static-session"

# Helper to split text into chunks
def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Ingest the initial document into the graph
def ingest_document(text: str):
    chunks = split_into_chunks(text)
    for chunk in chunks:
        client.graph.add(
            user_id=user_id,
            type="text",
            data=chunk
        )
    # print("Document ingested successfully.")

# Query information from the graph
def query(query_text: str):
    search_results = client.graph.search(
        user_id=user_id,
        query=query_text,
        scope="edges",
        limit=3
    )
    # print(search_results)
    if search_results.edges:
        return [edge.fact for edge in search_results.edges]
    else:
        return []
    
    

if __name__ == "__main__":
    with open("src/data/events.txt", "r", encoding="utf-8") as f:
        text = f.read()
    ingest_document(text)

    results = query("Which car does Maria has?")
    print("Question: Which car does Maria has?")
    # print("ChunksResults:")
    # for chunk in results:
    #     print(chunk)

    # Generate answer using LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    output_parser = StrOutputParser()
    
    qna_chain = create_qna_prompt() | llm | output_parser
    
    answer = qna_chain.invoke({
        "context": results,
        "question": "Which car does Maria has?"
    })
    
    answer = answer.strip()
    print("Answer:")
    print(answer)