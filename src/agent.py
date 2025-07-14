"""
Simple Insurance Timeline Agent
"""
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from src.timeline_tool import TimelineMapReduceTool

load_dotenv()

class InsuranceTimelineAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.tool = TimelineMapReduceTool()
        self.agent = initialize_agent(
            tools=[Tool(
                name="timeline_generator",
                description="Generate structured timeline from insurance events. Returns formatted timeline with dates and event types, no summaries.",
                func=self.tool.run
            )],
            llm=self.llm,
            verbose=False
        )
    
    def process(self, text: str) -> str:
        # Use the tool directly to avoid agent interpretation
        return self.tool.run(text)

def create_agent() -> InsuranceTimelineAgent:
    """Create and return an insurance timeline agent"""
    return InsuranceTimelineAgent()