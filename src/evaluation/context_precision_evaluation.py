"""
Context Precision Evaluation using Ragas
"""
from datasets import Dataset
from ragas.metrics import context_precision
from ragas import evaluate
from typing import List, Dict, Any

def evaluate_context_precision(question: str, answer: str, contexts: List[str]) -> float:
    """
    Evaluate the context precision of retrieved context chunks.
    This measures whether relevant chunks are ranked higher in the results.
    
    Args:
        question (str): The original question
        answer (str): The generated answer
        contexts (List[str]): List of context documents retrieved (in order of ranking)
        
    Returns:
        float: Context precision score between 0 and 1
    """
    try:
        # Generate ground truth for precision evaluation
        ground_truth = _generate_ground_truth_for_precision(question, contexts)
        
        # Prepare data for Ragas evaluation
        data_samples = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        }
        
        # Create dataset
        dataset = Dataset.from_dict(data_samples)
        
        # Evaluate context precision
        score = evaluate(dataset, metrics=[context_precision])
        
        # Extract context precision score - handle both list and single value
        context_precision_score = score['context_precision']
        if isinstance(context_precision_score, list):
            context_precision_score = context_precision_score[0]  # Get first (and only) score
        
        return context_precision_score
        
    except Exception as e:
        print(f"Error evaluating context precision: {e}")
        return 0.0

def _generate_ground_truth_for_precision(question: str, contexts: List[str]) -> str:
    """
    Generate ground truth for context precision evaluation.
    This creates a ground truth that represents what should be answerable from the context.
    
    Args:
        question (str): The original question
        contexts (List[str]): List of context documents
        
    Returns:
        str: Generated ground truth answer
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        # Create LLM for generating ground truth
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        
        # Create prompt for generating ground truth for precision evaluation
        ground_truth_prompt = ChatPromptTemplate.from_template("""
        Based on the following context chunks (in order of retrieval), provide a comprehensive and accurate answer to the question.
        Your answer should be the ground truth - the best possible answer that can be derived from the given context chunks.
        Focus on information that is most relevant to answering the question.
        
        Question: {question}
        
        Context Chunks (in retrieval order):
        {context}
        
        Ground Truth Answer:
        """)
        
        # Combine context with numbering to show order
        numbered_context = "\n\n".join([f"Chunk {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        # Generate ground truth using LLM
        chain = ground_truth_prompt | llm
        ground_truth = chain.invoke({
            "question": question,
            "context": numbered_context
        })
        
        return ground_truth.content.strip()
        
    except Exception as e:
        print(f"Error generating ground truth for precision: {e}")
        # Fallback to basic extraction if LLM fails
        return _fallback_ground_truth_for_precision(question, contexts)

def _fallback_ground_truth_for_precision(question: str, contexts: List[str]) -> str:
    """
    Fallback method for generating ground truth when LLM is not available.
    
    Args:
        question (str): The original question
        contexts (List[str]): List of context documents
        
    Returns:
        str: Generated ground truth answer
    """
    combined_context = "\n".join(contexts)
    
    # Extract key information from context that relates to the question
    if "when" in question.lower() or "date" in question.lower():
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, combined_context)
        if dates:
            return f"The event occurred on {dates[0]}"
    
    if "what happened" in question.lower() or "incident" in question.lower():
        if "INCIDENT_OCCURRED" in combined_context:
            return "An incident occurred as described in the context"
    
    # Default ground truth based on available context
    return f"Based on the available context: {combined_context[:200]}..."

def get_context_precision_grade(score: float) -> int:
    """
    Convert context precision score to a numerical grade from 1 to 100.
    
    Args:
        score (float): Context precision score between 0 and 1
        
    Returns:
        int: Numerical grade from 1 to 100
    """
    # Convert score from 0-1 range to 1-100 range
    numerical_grade = int(score * 100)
    return max(1, numerical_grade)  # Ensure minimum grade is 1

def print_context_precision_result(question: str, answer: str, contexts: List[str]) -> float:
    """
    Evaluate and print the context precision result for a Q&A pair.
    
    Args:
        question (str): The original question
        answer (str): The generated answer
        contexts (List[str]): List of context documents used (in retrieval order)
    """
    # Evaluate context precision
    context_precision_score = evaluate_context_precision(question, answer, contexts)
    
    # Get grade
    grade = get_context_precision_grade(context_precision_score)
    
    # Print result
    print(f"Context Precision Grade: {grade}")
    
    return context_precision_score 