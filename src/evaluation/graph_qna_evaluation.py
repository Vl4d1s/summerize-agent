from src.tools.utils.answer_with_graph import answer_question_with_graph

if __name__ == "__main__":
    questions = [
        # Easy: direct fact
        "Tell me about Maria",
        # Medium: requires combining info
        "What were the main reasons for the increase in Maria's insurance premium during the policy renewal?",
        # Hard: requires synthesis or is not directly stated
        "How did subrogation efforts impact the total claims cost for Maria's policy?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i}: {question}")
        answer = answer_question_with_graph(question, with_evaluation=True)
        print(f"Answer: {answer}")
