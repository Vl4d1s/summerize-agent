#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.tools.timeline_tool import get_timeline_tool
from src.tools.qna_tool import get_qna_tool
from src.prompts import create_summary_timeline_agent_prompt, create_qna_agent_prompt
from src.classifier import classify_for_agents





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
            classification = classify_for_agents(user_question, ["summery", "qna"], "qna")
            
            # Route to appropriate agent
            if classification == "summery":
                print("🕐 Routing to Timeline Agent...")
                tools = get_timeline_tool(use_refine=False)
                agent = create_react_agent(llm=llm, tools=tools, prompt=create_summary_timeline_agent_prompt())
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
                result = agent_executor.invoke({"input": user_question})
            elif classification == "qna":
                print("🤖 Routing to QnA Agent...")
                tools = get_qna_tool()
                agent = create_react_agent(llm=llm, tools=tools, prompt=create_qna_agent_prompt())
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
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

