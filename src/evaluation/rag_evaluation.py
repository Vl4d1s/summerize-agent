from src.tools.utils.answer_with_rag import answer_question_with_rag

if __name__ == "__main__":
    questions = [
        # Easy: direct fact
        "When did Maria Rodriguez's auto insurance policy begin?",
        # Medium: requires combining info
        "What were the main reasons for the increase in Maria's insurance premium during the policy renewal?",
        # Hard: requires synthesis or is not directly stated
        "How did subrogation efforts impact the total claims cost for Maria's policy?",
        # Not answerable from context
        "What is Maria Rodriguez's driver's license number?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i}: {question}")
        answer = answer_question_with_rag(question, with_evaluation=True)
        print(f"Answer: {answer}")
