#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.timeline_tool import get_timeline_tools
from src.prompts import create_agent_prompt


def main():
    load_dotenv()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    tools = get_timeline_tools(use_refine=False)
    agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    result = agent_executor.invoke({"input": "Create a chronological timeline from the insurance events."})
    timeline = result["output"]

    print("\nGenerated Timeline:")
    print("=" * 50)
    print(timeline)

if __name__ == "__main__":
    main() 