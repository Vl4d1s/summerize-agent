"""
Simple MapReduce Timeline Tool
"""
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.prompts import MAP_PROMPT, REDUCE_PROMPT

class TimelineMapReduceTool:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        
    def run(self, text: str) -> str:
        # Split text into documents
        docs = self.splitter.create_documents([text])
        
        # Create map chain
        map_chain = LLMChain(llm=self.llm, prompt=MAP_PROMPT)
        
        # Create reduce chain
        reduce_chain = LLMChain(llm=self.llm, prompt=REDUCE_PROMPT)
        reduce_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="text"
        )
        
        # Create map-reduce chain
        mapreduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="text"
        )
        
        return mapreduce_chain.run(docs) 