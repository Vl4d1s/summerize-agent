#!/usr/bin/env python3
"""
Simple Insurance Timeline Agent
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from src.agent import create_agent

def main():
    # Load environment variables
    load_dotenv()
    
    # Create agent with flag to choose approach
    # use_refine=False: Uses map-reduce pattern (default)
    # use_refine=True: Uses refine pattern
    agent = create_agent(use_refine=False)
    
    # Load text from file
    loader = TextLoader('events.txt')
    documents = loader.load()
    
    # Process text
    text = documents[0].page_content
    timeline = agent.process(text)
    
    print(timeline)

if __name__ == "__main__":
    main() 