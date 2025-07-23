"""
QnA Tool using RAG Pipeline
"""
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from src.tools.utils.answer_with_rag import answer_question_with_rag

class QnAInput(BaseModel):
    """Input schema for QnA tool"""
    question: str = Field(description="The question to answer")

def get_qna_rag_tool():
    """Get QnA tool that accepts a question and returns an answer using RAG"""
    qna_tool = Tool(
        name="qna_rag",
        description="Answer questions about insurance events using RAG pipeline. Input should be a question string.",
        func=answer_question_with_rag,
        args_schema=QnAInput,
    )
    return [qna_tool] 