"""
Context Recall Evaluation using Ragas
"""
from datasets import Dataset
from ragas.metrics import context_recall
from ragas import evaluate
from typing import List, Dict, Any

def evaluate_context_recall(question: str, answer: str, contexts: List[str]) -> float:
    """
    Evaluate the context recall of retrieved context by generating ground truth from available context.
    
    Args:
        question (str): The original question
        answer (str): The generated answer
        contexts (List[str]): List of context documents retrieved
        
    Returns:
        float: Context recall score between 0 and 1
    """
    try:
        # Generate ground truth from the available context
        # For context recall, we use the retrieved context as the source of truth
        # and create a ground truth answer based on what should be answerable from this context
        ground_truth = _generate_ground_truth_from_context(question, contexts)
        
        # Prepare data for Ragas evaluation
        data_samples = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        }
        
        # Create dataset
        dataset = Dataset.from_dict(data_samples)
        
        # Evaluate context recall
        score = evaluate(dataset, metrics=[context_recall])
        
        # Extract context recall score - handle both list and single value
        context_recall_score = score['context_recall']
        if isinstance(context_recall_score, list):
            context_recall_score = context_recall_score[0]  # Get first (and only) score
        
        return context_recall_score
        
    except Exception as e:
        print(f"Error evaluating context recall: {e}")
        return 0.0

def _generate_ground_truth_from_context(question: str, contexts: List[str]) -> str:
    """
    Generate a proper ground truth answer using LLM based on the question and context.
    This follows Ragas best practices for context recall evaluation.
    
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
        
        # Create prompt for generating ground truth
        ground_truth_prompt = ChatPromptTemplate.from_template("""
        Based on the following context, provide a comprehensive and accurate answer to the question.
        Your answer should be the ground truth - the best possible answer that can be derived from the given context.
        
        Question: {question}
        
        Context:
        {context}
        
        Ground Truth Answer:
        """)
        
        # Combine context
        combined_context = "\n\n".join(contexts)
        
        # Generate ground truth using LLM
        chain = ground_truth_prompt | llm
        ground_truth = chain.invoke({
            "question": question,
            "context": combined_context
        })
        
        return ground_truth.content.strip()
        
    except Exception as e:
        print(f"Error generating ground truth with LLM: {e}")
        # Fallback to basic extraction if LLM fails
        return _fallback_ground_truth_generation(question, contexts)

def _fallback_ground_truth_generation(question: str, contexts: List[str]) -> str:
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

def get_context_recall_grade(score: float) -> int:
    """
    Convert context recall score to a numerical grade from 1 to 100.
    
    Args:
        score (float): Context recall score between 0 and 1
        
    Returns:
        int: Numerical grade from 1 to 100
    """
    # Convert score from 0-1 range to 1-100 range
    numerical_grade = int(score * 100)
    return max(1, numerical_grade)  # Ensure minimum grade is 1

def print_context_recall_result(question: str, answer: str, contexts: List[str]) -> float:
    """
    Evaluate and print the context recall result for a Q&A pair.
    
    Args:
        question (str): The original question
        answer (str): The generated answer
        contexts (List[str]): List of context documents used
    """
    # Evaluate context recall
    context_recall_score = evaluate_context_recall(question, answer, contexts)
    
    # Get grade
    grade = get_context_recall_grade(context_recall_score)
    
    # Print result
    print(f"Context Recall Grade: {grade}")
    
    return context_recall_score 