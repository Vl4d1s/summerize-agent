#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.qna_tool import get_qna_tools
from src.prompts import create_agent_prompt


def main():
    load_dotenv()
    
    print("🚀 Starting QnA Agent with RAG Pipeline")
    print("=" * 50)
    
    # Initialize LLM and tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    tools = get_qna_tools()
    
    # Create agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # Example questions to test the QnA agent
    test_questions = [
        "When did John Smith purchase his auto insurance?",
        "What was the claim number for John's accident?",
        "How much was the settlement check?",
        "When was the claim approved?",
        "What was John's annual premium after renewal?"
    ]
    
    print("\n🤖 Testing QnA Agent with sample questions:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Question {i}: {question}")
        print("-" * 40)
        
        result = agent_executor.invoke({"input": question})
        answer = result["output"]
        
        print(f"\n💡 Answer: {answer}")
        print("=" * 50)
    
    print("\n✅ QnA Agent testing completed!")
    
    # Interactive mode
    print("\n🎯 Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        user_question = input("\n❓ Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_question:
            print("-" * 40)
            result = agent_executor.invoke({"input": user_question})
            answer = result["output"]
            print(f"\n💡 Answer: {answer}")
            print("-" * 40)


if __name__ == "__main__":
    main() 