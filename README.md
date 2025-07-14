# Insurance Timeline Agent

A simple and clean LangChain agent that creates chronological timelines from insurance-related text using map-reduce or refine patterns.

## Features

- **LangChain Agent Framework**: Uses ReAct pattern for intelligent reasoning
- **Map-Reduce Pattern**: Processes text in chunks and combines results
- **Refine Pattern**: Iteratively refines timeline with new information
- **Chronological Processing**: Automatically sorts events by date
- **Event Type Classification**: Standardizes event types (POLICY_START, CLAIM_FILED, etc.)
- **Date Estimation**: Handles approximate dates intelligently
- **Simple API**: Clean functional interface with pattern selection

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Command Line
```bash
# Use map-reduce pattern (default)
python main.py

# Use refine pattern
python main.py --refine
```

### Programmatic Usage
```python
from src.agent import process_text

# Map-reduce pattern (default)
timeline = process_text("Your insurance text here")

# Refine pattern
timeline = process_text("Your insurance text here", use_refine=True)
```

## Architecture

### Processing Patterns:
- **Map-Reduce**: Splits text into chunks, extracts events from each chunk (map), then combines all events into final timeline (reduce)
- **Refine**: Processes first chunk to create initial timeline, then iteratively refines with each subsequent chunk

### Agent Framework:
- Uses LangChain's ReAct agent pattern
- Powered by GPT-4o-mini for optimal performance
- Tools are dynamically created based on selected pattern

## Example Output

```
2023-01-15 - POLICY_START - Auto insurance purchased
2023-02-01 - PAYMENT_MADE - First premium payment made
2023-03-11 - INCIDENT_REPORTED - Incident reported to insurance company
2023-03-12 - CLAIM_FILED - Claim officially filed with claim number CL-2023-0045
2023-03-15 - ADJUSTER_ASSIGNED - Adjuster assigned to the case
2023-03-16 - INVESTIGATION_STARTED - Investigation began to determine fault and assess damages
2023-04-02 - INVESTIGATION_COMPLETED - Investigation completed; John not at fault
2023-04-05 - CLAIM_APPROVED - Claim approved; settlement check issued
2024-01-15 - POLICY_RENEWAL - Policy automatically renewed with premium increase
```

## Project Structure

```
summerize-agent/
├── main.py                 # Entry point with pattern selection
├── events.txt             # Sample insurance text
├── requirements.txt       # Dependencies
└── src/
    ├── agent.py          # Agent with pattern selection
    ├── timeline_tool.py  # Map-reduce and refine implementations
    └── prompts.py        # Specialized prompts for each pattern
```

## Key Features

- **Pattern Selection**: Choose between map-reduce and refine based on your needs
- **Simple & Clean**: Minimal code using LangChain's proven patterns
- **No Conditionals**: Straightforward execution flow
- **No Error Handling**: Clean and direct processing
- **Efficient**: Optimized chunking for long documents