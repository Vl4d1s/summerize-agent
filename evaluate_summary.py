import sys
from src.agent import process_text
from ragas.evaluation import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

# 1. Hardcoded input and ground truth
input_text = """John Doe filed an insurance claim on January 1, 2023, after his car was damaged in an accident. The insurance company reviewed the claim and approved it on January 5, 2023. Payment was issued to John on January 10, 2023."""
ground_truth = """John Doe filed a claim on Jan 1, 2023. It was approved on Jan 5, and payment was issued on Jan 10, 2023."""

# 2. Generate summary with agent
summary = process_text(input_text)

# 3. Prepare data for Ragas
data = Dataset.from_dict({
    "question": [input_text],
    "answer": [summary],
    "contexts": [[input_text]],
    "ground_truth": [ground_truth],
})

# 4. Evaluate and print results
results = evaluate(data, metrics=[faithfulness, answer_relevancy, context_recall])
print("Ragas Evaluation Results:")
print(results) 