from src.tools.timeline_tool import create_refine_chain
from src.evaluation.answer_correctness import evaluate_answer_correctness

if __name__ == "__main__":
    # 1. Hardcoded input and ground truth
    agent_query = "create a chronological timeline from the insurance events"

    with open("src/data/ground_truth.txt", "r", encoding="utf-8") as f:
        ground_truth_summary = f.read()

    # 2. Generate summary with agent
    agent_summary = create_refine_chain()

    # 3. Evaluate and print results
    results = evaluate_answer_correctness(agent_query, agent_summary, ground_truth_summary)
    print("Ragas Evaluation Results:")
    print(results) 