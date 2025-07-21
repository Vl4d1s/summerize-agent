from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from src.tools.qna_tool import get_qna_tool
from src.prompts import create_qna_agent_prompt

def get_qna_agent(llm=None):
    """Create and return the QnA AgentExecutor. Optionally accept an LLM instance."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    tools = get_qna_tool()
    agent = create_react_agent(llm=llm, tools=tools, prompt=create_qna_agent_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor
