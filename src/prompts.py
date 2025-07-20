"""
Prompts for Insurance Timeline Agent
"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

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

def create_agent_prompt() -> PromptTemplate:
    """Create the agent prompt template for ReAct agent"""
    return PromptTemplate.from_template("""
You are an expert insurance analyst that helps users create chronological timelines from insurance-related text and answer questions using RAG.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}
""")

def create_summary_timeline_agent_prompt() -> PromptTemplate:
    """Create the agent prompt template for summary timeline agent"""
    return PromptTemplate.from_template("""
You are an insurance timeline agent. Your job is simple: always use the timeline tool to process any input and return the tool's result directly to the user.

You have access to the following tool:
{tools}

IMPORTANT: You must ALWAYS use the tool for every question. Do not try to answer questions yourself. Simply:
1. Call the tool with the user's input
2. Return the tool's result as your final answer

Use the following format:

Question: the input question you must answer
Thought: I need to use the timeline tool to process this input
Action: the action to take, should be [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now have the result from the tool
Final Answer: the result from the tool

Question: {input}
{agent_scratchpad}
""")

def create_qna_agent_prompt() -> PromptTemplate:
    """Create the agent prompt template for QnA agent"""
    return PromptTemplate.from_template("""
You are an insurance QnA agent. Your job is simple: always use the RAG tool to answer any question and return the tool's result directly to the user.

You have access to the following tool:
{tools}

IMPORTANT: You must ALWAYS use the tool for every question. Do not try to answer questions yourself. Simply:
1. Call the tool with the user's question
2. Return the tool's result as your final answer

Use the following format:

Question: the input question you must answer
Thought: I need to use the RAG tool to answer this question
Action: the action to take, should be [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now have the result from the tool
Final Answer: the result from the tool

Question: {input}
{agent_scratchpad}
""")

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

Combined chronological timeline:"""
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

def create_qna_prompt() -> ChatPromptTemplate:
    """Create the QnA prompt template for answering questions using RAG"""
    return ChatPromptTemplate.from_template("""
You are an expert insurance analyst. Answer the question based on the provided context from insurance documents.

Rules:
- Base your answer ONLY on the information provided in the context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
- Provide clear, concise answers
- Include specific details from the context when relevant (dates, amounts, claim numbers, etc.)
- Do not make assumptions or add information not present in the context

Context:
{context}

Question: {question}

Answer:"""
    ) 



