"""
Timeline Tools using Proper Map-Reduce and Refine Patterns
"""
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from src.prompts import create_map_prompt, create_reduce_prompt

def create_timeline_tool(use_refine: bool = False, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
    """Create timeline tool with proper pattern implementation"""
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    output_parser = StrOutputParser()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    if use_refine:
        return create_refine_tool(llm, output_parser, splitter)
    else:
        return create_mapreduce_tool(llm, output_parser, splitter)

def create_mapreduce_tool(llm, output_parser, splitter):
    """Create proper map-reduce tool"""
    map_chain = create_map_prompt() | llm | output_parser
    reduce_chain = create_reduce_prompt() | llm | output_parser
    
    def process_text(text: str) -> str:
        """Process text using proper map-reduce pattern"""
        # Split text into documents
        docs = splitter.create_documents([text])
        
        # Map step: process each document chunk
        map_results = []
        for doc in docs:
            result = map_chain.invoke({"text": doc.page_content})
            map_results.append(result)
        
        # Reduce step: combine all map results
        combined_text = "\n".join(map_results)
        final_result = reduce_chain.invoke({"text": combined_text})
        
        return final_result.strip()
    
    return process_text

def create_refine_tool(llm, output_parser, splitter):
    """Create proper refine tool"""
    from src.prompts import create_initial_refine_prompt, create_refine_prompt
    
    initial_chain = create_initial_refine_prompt() | llm | output_parser
    refine_chain = create_refine_prompt() | llm | output_parser
    
    def process_text(text: str) -> str:
        """Process text using proper refine pattern"""
        # Split text into documents
        docs = splitter.create_documents([text])
        
        # Initial step: process first document
        current_result = initial_chain.invoke({"text": docs[0].page_content})
        
        # Refine step: iteratively refine with remaining documents
        for doc in docs[1:]:
            current_result = refine_chain.invoke({
                "existing_timeline": current_result,
                "new_text": doc.page_content
            })
        
        return current_result.strip()
    
    return process_text 