#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from src.agent import process_text

def main():
    load_dotenv()
    
    loader = TextLoader('events.txt')
    documents = loader.load()

    text = documents[0].page_content
    timeline = process_text(text, use_refine=False)
    
    print("\nGenerated Timeline:")
    print("=" * 50)
    print(timeline)

if __name__ == "__main__":
    main() 