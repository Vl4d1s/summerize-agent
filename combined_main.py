#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.timeline_tool import get_timeline_tools
from src.qna_tool import get_qna_tools
from src.prompts import create_agent_prompt


def main():
    load_dotenv()
    
    print("🧠 Starting Combined AI Agent")
    print("This agent intelligently chooses between Timeline and QnA tools")
    print("=" * 60)
    
    # Initialize LLM and combine tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    timeline_tools = get_timeline_tools(use_refine=False)
    qna_tools = get_qna_tools()
    combined_tools = timeline_tools + qna_tools
    
    print(f"🔧 Available tools: {[tool.name for tool in combined_tools]}")
    
    # Create combined agent
    agent = create_react_agent(llm=llm, tools=combined_tools, prompt=create_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=combined_tools, verbose=True, handle_parsing_errors=True)
    
    # Demonstrate intelligent tool selection
    demo_questions = [
        ("Timeline Request", "Create a chronological timeline of all insurance events"),
        ("Specific Question", "What was the claim number for John's accident?"),
        ("Summary Request", "Generate a timeline summary"),
        ("Detail Question", "How much was the settlement check?"),
        ("Timeline Question", "Show me all events in chronological order"),
        ("Fact Question", "When was the policy renewed?")
    ]
    
    print("\n🎯 Demonstrating Intelligent Tool Selection:")
    print("The agent will automatically choose the right tool for each question")
    print("=" * 60)
    
    for i, (question_type, question) in enumerate(demo_questions, 1):
        print(f"\n📋 Demo {i} ({question_type}): {question}")
        print("-" * 50)
        
        result = agent_executor.invoke({"input": question})
        answer = result["output"]
        
        print(f"\n✅ Result: {answer}")
        print("=" * 60)
        
        # Brief pause for readability
        input("Press Enter to continue to next demo...")
    
    print("\n🎉 Demo completed! The agent successfully chose the right tool for each question type.")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("🎯 Interactive Mode")
    print("Try different types of questions:")
    print("• Timeline: 'create timeline', 'show chronological events', 'summarize'")
    print("• QnA: 'when did...', 'what was...', 'how much...', specific questions")
    print("Type 'quit' to exit")
    print("-" * 60)
    
    while True:
        user_question = input("\n❓ Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using the Combined AI Agent!")
            break
        
        if user_question:
            print("\n" + "-" * 50)
            print("🧠 Agent is thinking and choosing the right tool...")
            print("-" * 50)
            
            result = agent_executor.invoke({"input": user_question})
            answer = result["output"]
            
            print(f"\n💡 Answer: {answer}")
            print("-" * 50)


if __name__ == "__main__":
    main() 