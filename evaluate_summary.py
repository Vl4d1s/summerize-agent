from src.timeline_tool import create_refine_chain
from ragas.evaluation import evaluate
from ragas.metrics import answer_correctness
from datasets import Dataset

# 1. Hardcoded input and ground truth
agent_query = "create a chronological timeline from the insurance events"

with open("ground_truth.txt", "r", encoding="utf-8") as f:
    ground_truth_summary = f.read()

# 2. Generate summary with agent
agent_summary = create_refine_chain()

# 3. Prepare data for Ragas
data = Dataset.from_dict({
    "question": [agent_query],
    "answer": [agent_summary],
    "ground_truth": [ground_truth_summary],
})

# 4. Evaluate and print results
results = evaluate(data, metrics=[answer_correctness])
print("Ragas Evaluation Results:")
print(results) 