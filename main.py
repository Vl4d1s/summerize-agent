#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.classifier import classify_for_agents
from src.agents.timeline_summery.agent import get_timeline_agent
from src.agents.qna.agent import get_qna_agent





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
            # Classify the question
            classification = classify_for_agents(user_question, ["summery", "qna"], "qna")
            
            # Route to appropriate agent
            if classification == "summery":
                print("Routing to Timeline Agent...")
                timeline_agent = get_timeline_agent(llm=llm)
                result = timeline_agent.invoke({"input": user_question})
            elif classification == "qna":
                print("Routing to QnA Agent...")
                qna_agent = get_qna_agent(llm=llm)
                result = qna_agent.invoke({"input": user_question})
            else:
                print(f"\u274C Unknown classification: {classification}")
                continue
                
            answer = result["output"]
            print(f"\n💡 Answer: {answer}")
            print("-" * 50)


def main():
    load_dotenv()
    run_classifier_agent()


if __name__ == "__main__":
    main() 

