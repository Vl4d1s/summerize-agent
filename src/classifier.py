"""
Question Classifier for Agent Selection
"""
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def create_classifier_prompt() -> ChatPromptTemplate:
    """Create the classifier prompt template with few-shot examples and role-playing"""
    return ChatPromptTemplate.from_template("""
You are an expert question classifier working for a knowledge management system. Your role is to accurately categorize incoming questions to route them to the appropriate response handlers. You have years of experience in natural language processing and question analysis.

Available Options: {options}

CLASSIFICATION RULES:
- "summery": Choose this for questions requesting timelines, chronological sequences, overviews, summaries of events, or broad narratives of "what happened"
- "qna": Choose this for questions seeking specific facts, details, precise information, or direct answers to who/what/when/where/how/why questions

EXAMPLES:

Question: "What happened during the American Civil War?"
Classification: summery
Reasoning: Asks for an overview/summary of events

Question: "When did the American Civil War start?"
Classification: qna
Reasoning: Seeks a specific date/fact

Question: "Can you give me a timeline of the company's growth?"
Classification: summery
Reasoning: Requests chronological overview

Question: "Who is the CEO of the company?"
Classification: qna
Reasoning: Asks for specific person/information

Question: "Tell me about the history of artificial intelligence"
Classification: summery
Reasoning: Asks for broad overview/narrative

Question: "What does AI stand for?"
Classification: qna
Reasoning: Seeks specific definition/fact

NOW CLASSIFY THIS QUESTION:
Question: {question}

Think carefully about whether the question seeks a broad overview/timeline (summery) or specific factual information (qna).

Classification (return ONLY the option):""")


def classify_question(question: str, options: list[str]) -> str:
    """
    Classify a question into one of the provided options using LLM
    
    Args:
        question: The question to classify
        options: List of available options to choose from
        
    Returns:
        One of the options from the provided list
    """
    
    # Initialize LLM and parser
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    output_parser = StrOutputParser()
    
    # Create classification chain
    classifier_chain = create_classifier_prompt() | llm | output_parser
    
    # Get classification
    options_str = ", ".join(options)
    result = classifier_chain.invoke({
        "question": question,
        "options": options_str
    })
    
    # Clean the result and ensure it's in the options
    classification = result.strip().lower()
    
    # Validate the result is in options
    valid_options = [opt.lower() for opt in options]
    if classification in valid_options:
        # Return the original case version
        original_option = options[valid_options.index(classification)]
        print(f"✅ Classification result: {original_option}")
        return original_option
    else:
        # Fallback to first option if invalid response
        print(f"⚠️ Invalid classification '{classification}', defaulting to '{options[0]}'")
        return options[0]


def classify_for_agents(question: str) -> str:
    """
    Classify question for agent selection (summery or qna)
    
    Args:
        question: The question to classify
        
    Returns:
        Either "summery" or "qna"
    """
    return classify_question(question, ["summery", "qna"]) 