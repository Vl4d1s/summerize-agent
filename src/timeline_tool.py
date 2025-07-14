"""
Timeline Tools using Map-Reduce and Refine Chains
"""
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from src.prompts import create_map_prompt, create_reduce_prompt, create_initial_refine_prompt, create_refine_prompt

def create_mapreduce_chain(text: str) -> str:
    """Create timeline using map-reduce pattern"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    output_parser = StrOutputParser()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    
    map_chain = create_map_prompt() | llm | output_parser
    reduce_chain = create_reduce_prompt() | llm | output_parser
    
    map_results = []
    for doc in docs:
        result = map_chain.invoke({"text": doc.page_content})
        map_results.append(result)
    
    combined_text = "\n".join(map_results)
    final_result = reduce_chain.invoke({"text": combined_text})
    
    return final_result.strip()

def create_refine_chain(text: str) -> str:
    """Create timeline using refine pattern"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    output_parser = StrOutputParser()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    
    initial_chain = create_initial_refine_prompt() | llm | output_parser
    refine_chain = create_refine_prompt() | llm | output_parser
    
    current_result = initial_chain.invoke({"text": docs[0].page_content})
    
    for doc in docs[1:]:
        current_result = refine_chain.invoke({
            "existing_timeline": current_result,
            "new_text": doc.page_content
        })
    
    return current_result.strip()

def get_timeline_tools(use_refine: bool = False):
    """Get timeline tools based on flag"""
    
    chain_function = create_refine_chain if use_refine else create_mapreduce_chain
    tool_name = "refine_timeline" if use_refine else "mapreduce_timeline"
    tool_description = f"Create chronological timeline using {'refine' if use_refine else 'map-reduce'} pattern"
    
    timeline_tool = Tool(
        name=tool_name,
        description=tool_description,
        func=chain_function
    )
    
    return [timeline_tool] 