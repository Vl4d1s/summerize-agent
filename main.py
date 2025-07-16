#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.timeline_tool import get_timeline_tools
from src.qna_tool import get_qna_tools
from src.prompts import create_agent_prompt


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


def main():
    load_dotenv()
    
    print("🚀 Insurance AI Agents")
    print("=" * 50)
    print("1. Timeline Agent - Creates chronological timelines")
    print("2. QnA Agent - Answers questions using RAG")
    print("3. Both agents")
    print("=" * 50)
    
    choice = input("Select an agent (1/2/3): ").strip()
    
    if choice == "1":
        run_timeline_agent()
    elif choice == "2":
        run_qna_agent()
    elif choice == "3":
        print("\n📋 Running both agents:")
        run_timeline_agent()
        print("\n" + "=" * 70 + "\n")
        run_qna_agent()
    else:
        print("Invalid choice. Running Timeline Agent by default.")
        run_timeline_agent()


if __name__ == "__main__":
    main() 