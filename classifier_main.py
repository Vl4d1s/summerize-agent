#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.timeline_tool import get_timeline_tools
from src.qna_tool import get_qna_tools
from src.prompts import create_agent_prompt
from src.classifier import classify_for_agents


def main():
    load_dotenv()
    
    print("🎯 Starting Classifier-based AI Agent")
    print("This agent classifies questions first, then routes to the appropriate tool")
    print("=" * 70)
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    
    print("🔧 Available classifications: summery (Timeline Agent) | qna (QnA Agent)")
    
    # Demonstrate classification with various question types
    demo_questions = [
        ("Timeline Request", "Create a chronological timeline of all insurance events"),
        ("Summary Request", "Generate a timeline summary"),
        ("Overview Request", "Show me what happened chronologically"),
        ("Specific Question", "What was the claim number for John's accident?"),
        ("Detail Question", "How much was the settlement check?"),
        ("Date Question", "When was the adjuster assigned?"),
        ("Amount Question", "What was the annual premium?"),
        ("Process Question", "Show me the sequence of events"),
    ]
    
    print("\n🧪 Demonstrating Classification Logic:")
    print("Each question will be classified, then routed to the appropriate agent")
    print("=" * 70)
    
    for i, (question_type, question) in enumerate(demo_questions, 1):
        print(f"\n📋 Demo {i} ({question_type}):")
        print(f"Question: {question}")
        print("-" * 50)
        
        # Step 1: Classify the question
        classification = classify_for_agents(question)
        
        # Step 2: Route to appropriate agent based on classification
        if classification == "summery":
            print("🕐 Classification → Timeline Agent")
            tools = get_timeline_tools(use_refine=False)
            agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
        elif classification == "qna":
            print("🤖 Classification → QnA Agent")
            tools = get_qna_tools()
            agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
        else:
            print(f"❌ Unknown classification: {classification}")
            continue
        
        # Step 3: Process the question with the selected agent
        result = agent_executor.invoke({"input": question})
        answer = result["output"]
        
        print(f"\n✅ Result: {answer}")
        print("=" * 70)
        
        # Brief pause for readability
        input("Press Enter to continue to next demo...")
    
    print("\n🎉 Classification demo completed!")
    print("The agent successfully classified each question and routed to the correct tool.")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("🎯 Interactive Classification Mode")
    print("Your questions will be classified first, then processed by the appropriate agent")
    print("Classification options:")
    print("• 'summery' → Timeline Agent (for timelines, chronology, summaries)")
    print("• 'qna' → QnA Agent (for specific questions, facts, details)")
    print("Type 'quit' to exit")
    print("-" * 70)
    
    while True:
        user_question = input("\n❓ Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using the Classifier-based AI Agent!")
            break
        
        if user_question:
            print("\n" + "-" * 50)
            print("🔍 Step 1: Classifying your question...")
            
            # Classify the question
            classification = classify_for_agents(user_question)
            
            # Route to appropriate agent
            if classification == "summery":
                print("🕐 Step 2: Routing to Timeline Agent...")
                tools = get_timeline_tools(use_refine=False)
                agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
            elif classification == "qna":
                print("🤖 Step 2: Routing to QnA Agent...")
                tools = get_qna_tools()
                agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
            else:
                print(f"❌ Unknown classification: {classification}")
                continue
            
            print("⚙️ Step 3: Processing with selected agent...")
            result = agent_executor.invoke({"input": user_question})
            answer = result["output"]
            
            print(f"\n💡 Final Answer: {answer}")
            print("-" * 50)


if __name__ == "__main__":
    main() 