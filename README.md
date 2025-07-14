# Insurance Timeline Agent

A simple LangChain agent that converts insurance event text into structured timelines.

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```

## Usage

```bash
python main.py
```

This will process the text in `events.txt` and output a timeline.

## Programmatic Usage

```python
from src.agent import create_agent

agent = create_agent()
timeline = agent.process(your_text)
print(timeline)
```

## Output Format

The agent generates structured timeline events:

```
2023-01-01 - POLICY_START - Insurance policy purchased
2023-03-05 - CLAIM_FILED - Claim filed
2023-03-20 - CLAIM_APPROVED - Claim approved
2023-04-05 - SETTLEMENT_ISSUED - Settlement check issued
```

## Event Types

- **Policy Events**: `POLICY_START`, `POLICY_RENEWAL`, `POLICY_CANCELED`
- **Claim Events**: `CLAIM_FILED`, `CLAIM_APPROVED`, `CLAIM_DENIED`
- **Payment Events**: `PAYMENT_MADE`, `PAYMENT_MISSED`
- **Investigation**: `INSPECTION`, `ADJUSTER_VISIT`
- **Incidents**: `INCIDENT_REPORTED`, `SETTLEMENT_ISSUED`