from tools.utils.answer_with_rag import answer_question_with_rag

if __name__ == "__main__":
    questions = [
        # Easy: direct fact
        "When did John Smith purchase auto insurance?",
        # Medium: requires combining info
        "What was the outcome of John's car accident claim?",
        # Hard: requires synthesis or is not directly stated
        "Why did John's insurance premium increase in 2024?",
        # Not answerable from context
        "What is John's driver's license number?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i}: {question}")
        answer = answer_question_with_rag(question, with_evaluation=True)
        print(f"Answer: {answer}")
