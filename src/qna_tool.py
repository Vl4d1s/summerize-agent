"""
QnA Tool using RAG Pipeline
"""
import os
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from src.prompts import create_qna_prompt

class QnAInput(BaseModel):
    """Input schema for QnA tool"""
    question: str = Field(description="The question to answer")

def create_vector_store():
    """Create and populate vector store with events.txt content"""
    print("📚 Loading text from events.txt...")
    
    # Load text from events.txt
    with open("events.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    print(f"📄 Loaded {len(text)} characters from events.txt")
    
    # Split into chunks
    print("✂️ Splitting text into chunks (size=200, overlap=30)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=30
    )
    docs = splitter.create_documents([text])
    print(f"📑 Created {len(docs)} chunks")
    
    # Create embeddings
    print("🔮 Creating OpenAI embeddings...")
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    print("💾 Creating Chroma vector store...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Vector store created successfully")
    
    return vector_store

def answer_question_with_rag(question: str) -> str:
    """Answer question using RAG pipeline"""
    print(f"\n❓ Processing question: '{question}'")
    
    # Create or load vector store
    print("🔍 Setting up vector store...")
    embeddings = OpenAIEmbeddings()
    
    # Check if vector store exists
    if os.path.exists("./chroma_db"):
        print("📂 Loading existing vector store...")
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        print("🆕 Creating new vector store...")
        vector_store = create_vector_store()
    
    # Embed question and retrieve relevant chunks
    print("🔎 Searching for relevant chunks...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)
    
    print(f"📝 Found {len(relevant_docs)} relevant chunks:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"   Chunk {i}: {doc.page_content[:100]}...")
    
    # Combine chunks into context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate answer using LLM
    print("🤖 Generating answer with LLM...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    output_parser = StrOutputParser()
    
    qna_chain = create_qna_prompt() | llm | output_parser
    
    answer = qna_chain.invoke({
        "context": context,
        "question": question
    })
    
    print("✅ Answer generated successfully")
    return answer.strip()

def get_qna_tools():
    """Get QnA tool that accepts a question and returns an answer using RAG"""
    qna_tool = Tool(
        name="qna_rag",
        description="Answer questions about insurance events using RAG pipeline. Input should be a question string.",
        func=answer_question_with_rag,
        args_schema=QnAInput,
    )
    return [qna_tool] 