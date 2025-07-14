# Insurance Events Timeline Agent

## Project Overview

A LangChain-based agent that processes insurance customer event sequences and generates structured timeline summaries with dates.

## Architecture

```
Plain Text Events → Agent → MapReduce/Refine Tool → Structured Timeline Summary
```

## Core Components

### 1. Agent Configuration
- **Framework**: LangChain
- **Model**: GPT-4 or equivalent
- **Tools**: Single tool (MapReduce OR Refine)
- **Input**: Plain text event sequences
- **Output**: Structured timeline with dates

### 2. Tool Implementation

#### Option A: MapReduce Approach
```python
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain

def create_mapreduce_tool():
    """
    Map: Extract individual events with dates
    Reduce: Combine into chronological timeline
    """
    map_prompt = """
    Extract insurance events with dates from this text:
    {text}
    
    Format: DATE - EVENT_TYPE - DESCRIPTION
    """
    
    reduce_prompt = """
    Combine these events into a chronological timeline:
    {text}
    
    Output as structured timeline with dates.
    """
```

#### Option B: Refine Approach
```python
from langchain.chains.summarize import load_summarize_chain

def create_refine_tool():
    """
    Iteratively refine timeline summary
    """
    refine_prompt = """
    Current timeline: {existing_answer}
    
    New events to integrate: {text}
    
    Refine the timeline to include new events chronologically.
    """
```

## Implementation Plan

### Phase 1: Core Setup
1. **Environment Setup**
   ```bash
   pip install langchain openai python-dotenv
   ```

2. **Project Structure**
   ```
   insurance_agent/
   ├── src/
   │   ├── agent.py
   │   ├── tools/
   │   │   └── timeline_tool.py
   │   └── prompts/
   │       └── prompt_templates.py
   ├── tests/
   ├── examples/
   └── requirements.txt
   ```

### Phase 2: Agent Development

#### 1. Prompt Engineering with Few-Shot Examples
```python
FEW_SHOT_EXAMPLES = """
Example 1:
Input: "John filed claim on 2023-05-15 for car accident. Policy renewed 2023-01-01. Claim approved 2023-05-20."
Output:
2023-01-01 - POLICY_RENEWAL - Policy renewed
2023-05-15 - CLAIM_FILED - Car accident claim filed
2023-05-20 - CLAIM_APPROVED - Claim approved

Example 2:
Input: "Premium payment missed February 2023. Policy canceled March 15, 2023. Reinstatement request April 1, 2023."
Output:
2023-02-01 - PAYMENT_MISSED - Premium payment missed
2023-03-15 - POLICY_CANCELED - Policy canceled
2023-04-01 - REINSTATEMENT_REQUEST - Reinstatement request submitted
"""
```

#### 2. Role-Playing Prompt
```python
ROLE_PROMPT = """
You are an expert insurance claims analyst with 20+ years of experience.
Your specialty is organizing customer event histories into clear, chronological timelines.

Task: Process the insurance customer's event sequence and create a structured timeline.

Guidelines:
- Extract all events with dates
- Categorize events (CLAIM, POLICY, PAYMENT, etc.)
- Sort chronologically
- Use format: YYYY-MM-DD - CATEGORY - DESCRIPTION
"""
```

### Phase 3: Tool Implementation

#### Core Tool Class
```python
class TimelineGeneratorTool:
    def __init__(self, approach="mapreduce"):
        self.approach = approach
        self.chain = self._create_chain()
    
    def _create_chain(self):
        if self.approach == "mapreduce":
            return self._create_mapreduce_chain()
        else:
            return self._create_refine_chain()
    
    def process_events(self, text: str) -> str:
        return self.chain.run(text)
```

### Phase 4: Agent Assembly

#### Simple Agent Configuration
```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

def create_insurance_agent():
    llm = OpenAI(temperature=0)
    
    timeline_tool = Tool(
        name="Timeline Generator",
        description="Processes insurance events and creates structured timeline",
        func=TimelineGeneratorTool().process_events
    )
    
    agent = initialize_agent(
        tools=[timeline_tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    
    return agent
```

## Usage Example

### Input
```
"Customer John Doe purchased auto insurance policy on January 1, 2023. 
First premium payment made January 5, 2023. Minor accident reported March 15, 2023. 
Claim filed March 16, 2023. Claim investigation completed April 2, 2023. 
Claim approved and payment issued April 5, 2023."
```

### Expected Output
```
2023-01-01 - POLICY_PURCHASE - Auto insurance policy purchased
2023-01-05 - PREMIUM_PAYMENT - First premium payment made
2023-03-15 - INCIDENT_REPORTED - Minor accident reported
2023-03-16 - CLAIM_FILED - Claim filed
2023-04-02 - INVESTIGATION_COMPLETED - Claim investigation completed
2023-04-05 - CLAIM_APPROVED - Claim approved and payment issued
```

## Best Practices

### 1. Prompt Engineering
- Use specific role-playing context
- Include few-shot examples
- Clear output format specifications
- Handle edge cases in prompts

### 2. Error Handling
```python
def safe_process_events(text: str) -> str:
    try:
        return timeline_tool.process_events(text)
    except Exception as e:
        return f"Error processing events: {str(e)}"
```

## Development Checklist

- [ ] Set up development environment
- [ ] Implement MapReduce/Refine tool
- [ ] Create prompt templates with few-shot examples
- [ ] Build agent with single tool
- [ ] Test with sample insurance events
- [ ] Add error handling and validation
- [ ] Document API and usage

## Dependencies

```
langchain>=0.0.200
openai>=0.27.0
python-dotenv>=1.0.0
```

## Notes

- Keep implementation simple and focused
- Single responsibility: text → structured timeline
- Emphasize clean, readable code
- Follow LangChain best practices
- Maintain flexibility for future enhancements