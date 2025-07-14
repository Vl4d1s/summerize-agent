from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.timeline_tool import get_timeline_tools
from src.prompts import create_agent_prompt

def create_agent(use_refine: bool = False, model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> AgentExecutor:
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    tools = get_timeline_tools(use_refine=use_refine)

    agent = create_react_agent(llm=llm, tools=tools, prompt=create_agent_prompt())
    
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

def process_text(text: str, use_refine: bool = False) -> str:
    agent_executor = create_agent(use_refine=use_refine)
    query = f"Create a chronological timeline from this insurance text: {text}"
    result = agent_executor.invoke({"input": query})
    
    return result["output"]