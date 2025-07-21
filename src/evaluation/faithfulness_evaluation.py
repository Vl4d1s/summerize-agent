"""
Faithfulness Evaluation using Ragas
"""
from datasets import Dataset
from ragas.metrics import faithfulness
from ragas import evaluate
from typing import List, Dict, Any

def evaluate_faithfulness(question: str, answer: str, contexts: List[str]) -> float:
    """
    Evaluate the faithfulness of a generated answer against the provided context.
    
    Args:
        question (str): The original question
        answer (str): The generated answer to evaluate
        contexts (List[str]): List of context documents used to generate the answer
        
    Returns:
        float: Faithfulness score between 0 and 1
    """
    try:
        # Prepare data for Ragas evaluation
        data_samples = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts]
        }
        
        # Create dataset
        dataset = Dataset.from_dict(data_samples)
        
        # Evaluate faithfulness
        score = evaluate(dataset, metrics=[faithfulness])
        
        # Extract faithfulness score - handle both list and single value
        faithfulness_score = score['faithfulness']
        if isinstance(faithfulness_score, list):
            faithfulness_score = faithfulness_score[0]  # Get first (and only) score
        
        return faithfulness_score
        
    except Exception as e:
        print(f"Error evaluating faithfulness: {e}")
        return 0.0

def get_faithfulness_grade(score: float) -> int:
    """
    Convert faithfulness score to a numerical grade from 1 to 100.
    
    Args:
        score (float): Faithfulness score between 0 and 1
        
    Returns:
        int: Numerical grade from 1 to 100
    """
    # Convert score from 0-1 range to 1-100 range
    numerical_grade = int(score * 100)
    return max(1, numerical_grade)  # Ensure minimum grade is 1

def print_faithfulness_result(question: str, answer: str, contexts: List[str]) -> None:
    """
    Evaluate and print the faithfulness result for a Q&A pair.
    
    Args:
        question (str): The original question
        answer (str): The generated answer
        contexts (List[str]): List of context documents used
    """
    # Evaluate faithfulness
    faithfulness_score = evaluate_faithfulness(question, answer, contexts)
    
    # Get grade
    grade = get_faithfulness_grade(faithfulness_score)
    
    # Print result
    print(f"Faithfulness Grade: {grade}")
    
    return faithfulness_score 