# Insurance Timeline Agent

A simple LangChain agent that converts insurance event text into structured timelines using MapReduce.

## Features

- **Simple Architecture**: One agent, one tool, clean flow
- **MapReduce Processing**: Handles texts of any length
- **Few-Shot Prompting**: Consistent output format
- **Role-Playing**: Expert insurance analyst persona

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

```python
from langchain_community.document_loaders import TextLoader
from src.agent import create_agent

# Load text from file
loader = TextLoader("events.txt")
documents = loader.load()
text = documents[0].page_content

# Create agent and process
agent = create_agent()
timeline = agent.process(text)
print(timeline)
```

## Quick Start

1. Add your insurance events to `events.txt`
2. Run the agent:

```bash
python main.py
```

## Output Format

```
2023-01-01 - POLICY_START - Insurance policy purchased
2023-03-05 - CLAIM_FILED - Claim filed
2023-03-20 - CLAIM_APPROVED - Claim approved
```

## Project Structure

```
├── src/
│   ├── agent.py          # Main agent
│   ├── timeline_tool.py  # MapReduce tool
│   └── prompts.py        # Few-shot prompts
├── main.py               # Usage example
└── requirements.txt      # Dependencies
```

## Event Types

- POLICY_START, POLICY_RENEWAL, POLICY_CANCELED
- CLAIM_FILED, CLAIM_APPROVED, CLAIM_DENIED
- PAYMENT_MADE, PAYMENT_MISSED
- INSPECTION, ADJUSTER_VISIT
- INCIDENT_REPORTED, SETTLEMENT_ISSUED