"""
Simple Insurance Timeline Agent
"""
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from src.timeline_tool import create_timeline_tool

class InsuranceTimelineAgent:
    """Simple insurance timeline agent"""
    
    def __init__(self, use_refine: bool = False, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """Initialize the agent"""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.tool = create_timeline_tool(use_refine=use_refine, model_name=model_name, temperature=temperature)
        self.processor = RunnableLambda(self._process_text)
    
    def _process_text(self, text: str) -> str:
        """Process text through the timeline tool"""
        return self.tool(text)
    
    def process(self, text: str) -> str:
        """Process insurance text to generate timeline"""
        return self.processor.invoke(text)

def create_agent(use_refine: bool = False, model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> InsuranceTimelineAgent:
    """Create and return an insurance timeline agent"""
    return InsuranceTimelineAgent(use_refine=use_refine, model_name=model_name, temperature=temperature)