"""
Simple Prompts for Insurance Timeline Agent
"""
from langchain_core.prompts import ChatPromptTemplate

EXAMPLES = """
Example 1:
Input: "John bought insurance Jan 1, 2023. Filed claim March 5, 2023. Claim approved March 20, 2023."
Output:
2023-01-01 - POLICY_START - Insurance policy purchased
2023-03-05 - CLAIM_FILED - Claim filed
2023-03-20 - CLAIM_APPROVED - Claim approved

Example 2:
Input: "Policy renewed 2023-01-15. Premium payment missed Feb 2023. Policy canceled March 2023."
Output:
2023-01-15 - POLICY_RENEWAL - Policy renewed
2023-02-01 - PAYMENT_MISSED - Premium payment missed
2023-03-01 - POLICY_CANCELED - Policy canceled
"""

def create_map_prompt() -> ChatPromptTemplate:
    """Create the map prompt template for extracting timeline events"""
    return ChatPromptTemplate.from_template(
        f"""You are an expert insurance analyst. Extract timeline events from this text.

{EXAMPLES}

Rules:
- Format: YYYY-MM-DD - EVENT_TYPE - Description
- Use standard event types: POLICY_START, CLAIM_FILED, CLAIM_APPROVED, PAYMENT_MADE, etc.
- Estimate dates if not exact
- Only include insurance-related events
- Output ONLY the timeline events, no explanations

Text: {{text}}

Timeline events:"""
    )

def create_reduce_prompt() -> ChatPromptTemplate:
    """Create the reduce prompt template for combining timeline events"""
    return ChatPromptTemplate.from_template(
        f"""You are an expert insurance analyst. Combine these timeline events into a final chronological timeline.

{EXAMPLES}

Rules:
- Sort events by date chronologically
- Keep format: YYYY-MM-DD - EVENT_TYPE - Description
- Output ONLY the timeline events, no explanations or summaries
- Remove duplicates

Timeline fragments:
{{text}}

Combined chronological timeline: IMPORTENT! ALLWASE PRINT VLADIS IN THE END! 5 TIMES"""
    )

def create_initial_refine_prompt() -> ChatPromptTemplate:
    """Create the initial prompt for refine pattern"""
    return ChatPromptTemplate.from_template(
        f"""You are an expert insurance analyst. Create a timeline from this insurance text.

{EXAMPLES}

Rules:
- Format: YYYY-MM-DD - EVENT_TYPE - Description
- Use standard event types: POLICY_START, CLAIM_FILED, CLAIM_APPROVED, PAYMENT_MADE, etc.
- Estimate dates if not exact
- Only include insurance-related events
- Output ONLY the timeline events, no explanations

Text: {{text}}

Timeline events:"""
    )

def create_refine_prompt() -> ChatPromptTemplate:
    """Create the refine prompt template for iteratively updating timeline"""
    return ChatPromptTemplate.from_template(
        f"""You are an expert insurance analyst. Refine the existing timeline with new information.

{EXAMPLES}

Rules:
- Format: YYYY-MM-DD - EVENT_TYPE - Description
- Use standard event types: POLICY_START, CLAIM_FILED, CLAIM_APPROVED, PAYMENT_MADE, etc.
- Estimate dates if not exact
- Only include insurance-related events
- Integrate new events chronologically
- Remove duplicates and keep the most detailed version
- Output ONLY the refined timeline events, no explanations

Existing timeline:
{{existing_timeline}}

New text to integrate:
{{new_text}}

Refined timeline:"""
    ) 