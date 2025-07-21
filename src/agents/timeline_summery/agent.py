from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.tools.timeline_tool import get_timeline_tool
from src.agents.timeline_summery.prompt import create_summary_timeline_agent_prompt

def get_timeline_agent(llm=None):
    """Create and return the Timeline AgentExecutor. Optionally accept an LLM instance."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    tools = get_timeline_tool(use_refine=False)
    agent = create_react_agent(llm=llm, tools=tools, prompt=create_summary_timeline_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor
