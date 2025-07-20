#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.timeline_tool import get_timeline_tools
from src.qna_tool import get_qna_tools
from src.prompts import create_agent_prompt
from src.classifier import classify_for_agents


def run_timeline_agent():
    """Run the Timeline Agent"""
    print("🕐 Starting Timeline Agent")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    tools = get_timeline_tools(use_refine=False)
    agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    result = agent_executor.invoke({"input": "Create a chronological timeline from the insurance events."})
    timeline = result["output"]

    print("\n📅 Generated Timeline:")
    print("=" * 50)
    print(timeline)


def run_qna_agent():
    """Run the QnA Agent"""
    print("🤖 Starting QnA Agent with RAG Pipeline")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    tools = get_qna_tools()
    agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    
    # Ask a sample question
    question = "When did John Smith purchase his auto insurance and what was the premium?"
    print(f"\n❓ Sample Question: {question}")
    print("-" * 40)
    
    result = agent_executor.invoke({"input": question})
    answer = result["output"]
    
    print(f"\n💡 Answer: {answer}")


def run_combined_agent():
    """Run the Combined Agent with both Timeline and QnA tools"""
    print("🔄 Starting Combined Agent (Timeline + QnA)")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    
    # Combine both tool sets
    timeline_tools = get_timeline_tools(use_refine=False)
    qna_tools = get_qna_tools()
    combined_tools = timeline_tools + qna_tools
    
    agent = create_react_agent(llm=llm, tools=combined_tools, prompt=create_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=combined_tools, verbose=True, handle_parsing_errors=True)
    
    # Test with different types of questions
    test_cases = [
        "Create a chronological timeline from the insurance events",
        "When was John's claim approved?",
        "Generate a summary timeline of all events",
        "What was the settlement amount for the accident?"
    ]
    
    print("\n🧪 Testing Combined Agent with different question types:")
    print("=" * 60)
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {question}")
        print("-" * 50)
        
        result = agent_executor.invoke({"input": question})
        answer = result["output"]
        
        print(f"\n💡 Result: {answer}")
        print("=" * 60)
    
    print("\n✅ Combined Agent testing completed!")
    
    # Interactive mode
    print("\n🎯 Interactive Mode - The agent will choose the right tool!")
    print("Ask for timelines/summaries or specific questions")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        user_question = input("\n❓ Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_question:
            print("-" * 50)
            result = agent_executor.invoke({"input": user_question})
            answer = result["output"]
            print(f"\n💡 Answer: {answer}")
            print("-" * 50)


def run_classifier_agent():
    """Run the Classifier-based Agent that pre-classifies questions"""
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    
    print("Type 'quit' to exit")
    
    while True:
        user_question = input("\n❓ Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_question:
            print("\n" + "-" * 50)
            
            # Classify the question
            classification = classify_for_agents(user_question)
            
            # Route to appropriate agent
            if classification == "summery":
                print("🕐 Routing to Timeline Agent...")
                tools = get_timeline_tools(use_refine=False)
                agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
                result = agent_executor.invoke({"input": user_question})
            elif classification == "qna":
                print("🤖 Routing to QnA Agent...")
                tools = get_qna_tools()
                agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
                result = agent_executor.invoke({"input": user_question})
            else:
                print(f"❌ Unknown classification: {classification}")
                continue
                
            answer = result["output"]
            print(f"\n💡 Answer: {answer}")
            print("-" * 50)


def main():
    load_dotenv()
    run_classifier_agent()


if __name__ == "__main__":
    main() 










    # print("🚀 Insurance AI Agents")
    # print("=" * 50)
    # print("1. Timeline Agent - Creates chronological timelines")
    # print("2. QnA Agent - Answers questions using RAG")
    # print("3. Combined Agent - Intelligently chooses the right tool")
    # print("4. Classifier Agent - Pre-classifies questions then routes to agents")
    # print("5. Run all agents separately")
    # print("=" * 50)
    
    # choice = input("Select an agent (1/2/3/4/5): ").strip()
    
    # if choice == "1":
    #     run_timeline_agent()
    # elif choice == "2":
    #     run_qna_agent()
    # elif choice == "3":
    #     run_combined_agent()
    # elif choice == "4":
    #     run_classifier_agent()
    # elif choice == "5":
    #     print("\n📋 Running all agents:")
    #     run_timeline_agent()
    #     print("\n" + "=" * 70 + "\n")
    #     run_qna_agent()
    #     print("\n" + "=" * 70 + "\n")
    #     run_combined_agent()
    #     print("\n" + "=" * 70 + "\n")
    #     run_classifier_agent()
    # else:
    #     print("Invalid choice. Running Classifier Agent by default.")
    #     run_classifier_agent()