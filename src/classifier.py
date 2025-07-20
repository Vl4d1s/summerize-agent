"""
Question Classifier for Agent Selection
"""
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def create_classifier_prompt() -> ChatPromptTemplate:
    """Create the classifier prompt template"""
    return ChatPromptTemplate.from_template("""
You are a question classifier. Given a question and a list of options, you must return EXACTLY ONE of the provided options without any additional text, explanation, or formatting.

Available Options: {options}

Your task is to classify the question into the most appropriate option:
- If the question asks for a timeline, chronological order, summary of events, or overview of what happened, choose "summery"
- If the question asks for specific information, details, facts, dates, amounts, or answers to "who/what/when/where/how" questions, choose "qna"

Question: {question}

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