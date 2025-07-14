#!/usr/bin/env python3
"""
Simple usage example for Insurance Timeline Agent using document loader
"""
from langchain_community.document_loaders import TextLoader
from src.agent import create_agent

def main():
    # Load text from file using document loader
    loader = TextLoader("events.txt")
    documents = loader.load()
    
    # Extract text from the loaded document
    text = documents[0].page_content
    
    print("Insurance Timeline Agent")
    print("=" * 50)
    print(f"Loading from: events.txt")
    print(f"Document loaded: {len(text)} characters")
    print("\nProcessing...")
    
    # Create agent and process text
    agent = create_agent()
    timeline = agent.process(text)
    
    print("\nGenerated Timeline:")
    print("-" * 30)
    print(timeline)

if __name__ == "__main__":
    main() 