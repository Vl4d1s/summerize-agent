import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from src.prompts import create_qna_prompt
from src.evaluation.faithfulness_evaluation import print_faithfulness_result
from src.evaluation.context_recall_evaluation import print_context_recall_result
from src.evaluation.context_precision_evaluation import print_context_precision_result

def create_vector_store():
    """Create and populate vector store with events.txt content"""
    with open("src/data/events.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=30
    )
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store


def getAnswerForRetrieval(question: str) -> str:
    """Answer the question using the full content of ground_truth.txt as context."""
    with open("src/data/ground_truth.txt", "r", encoding="utf-8") as f:
        context = f.read().strip()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    output_parser = StrOutputParser()
    qna_chain = create_qna_prompt() | llm | output_parser
    answer = qna_chain.invoke({
        "context": context,
        "question": question
    })
    answer = answer.strip()
    return answer


def answer_question_with_rag(question: str , with_evaluation: bool = False,answer_retrieval: bool = False) -> str:
    """Answer question using RAG pipeline"""
    embeddings = OpenAIEmbeddings()
    if os.path.exists("./chroma_db"):
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        vector_store = create_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retrival_context = getAnswerForRetrieval(question) if answer_retrieval else question
    relevant_docs = retriever.invoke(retrival_context)
    print("Retrieved Context:")    
    for doc in relevant_docs:
        print("--------------------------------")
        print(doc.page_content.strip())
    print("--------------------------------")
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
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
