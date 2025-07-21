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
from src.evaluation.faithfulness_evaluation import print_faithfulness_result
from src.evaluation.context_recall_evaluation import print_context_recall_result
from src.evaluation.context_precision_evaluation import print_context_precision_result

class QnAInput(BaseModel):
    """Input schema for QnA tool"""
    question: str = Field(description="The question to answer")

def create_vector_store():
    """Create and populate vector store with events.txt content"""
    
    # Load text from events.txt
    with open("src/data/events.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=30
    )
    docs = splitter.create_documents([text])
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vector_store

def answer_question_with_rag(question: str , with_evaluation: bool = False) -> str:
    """Answer question using RAG pipeline"""
    
    # Create or load vector store
    embeddings = OpenAIEmbeddings()
    
    # Check if vector store exists
    if os.path.exists("./chroma_db"):
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        vector_store = create_vector_store()
    
    # Embed question and retrieve relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)
    
    
    # Combine chunks into context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate answer using LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    output_parser = StrOutputParser()
    
    qna_chain = create_qna_prompt() | llm | output_parser
    
    answer = qna_chain.invoke({
        "context": context,
        "question": question
    })
    
    answer = answer.strip()
    
    if with_evaluation:
        contexts_list = [doc.page_content for doc in relevant_docs]
        print_context_recall_result(question, answer, contexts_list)
        print_faithfulness_result(question, answer, contexts_list)
        print_context_precision_result(question, answer, contexts_list)
    
    return answer

def get_qna_tool():
    """Get QnA tool that accepts a question and returns an answer using RAG"""
    qna_tool = Tool(
        name="qna_rag",
        description="Answer questions about insurance events using RAG pipeline. Input should be a question string.",
        func=answer_question_with_rag,
        args_schema=QnAInput,
    )
    return [qna_tool] 