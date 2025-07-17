# Insurance AI Agents

A collection of simple and clean LangChain agents for insurance document processing, including timeline generation and question-answering with RAG (Retrieval Augmented Generation).

## Features

### Timeline Agent

- **LangChain Agent Framework**: Uses ReAct pattern for intelligent reasoning
- **Map-Reduce Pattern**: Processes text in chunks and combines results
- **Refine Pattern**: Iteratively refines timeline with new information
- **Chronological Processing**: Automatically sorts events by date
- **Event Type Classification**: Standardizes event types (POLICY_START, CLAIM_FILED, etc.)
- **Date Estimation**: Handles approximate dates intelligently

### QnA Agent

- **RAG Pipeline**: Retrieval Augmented Generation for accurate question answering
- **Vector Search**: Uses Chroma DB with OpenAI embeddings for semantic search
- **Context-Aware**: Answers questions based on relevant document chunks
- **Interactive Mode**: Real-time question answering interface
- **Chunk Processing**: Splits documents into 200-character chunks with 30-character overlap

### Combined Agent

- **Intelligent Tool Selection**: Automatically chooses between Timeline and QnA tools
- **Context Understanding**: Recognizes timeline requests vs. specific questions
- **Unified Interface**: Single agent that handles all types of queries
- **Demo Mode**: Shows how the agent selects tools for different question types

### Classifier Agent

- **Pre-Classification**: Explicitly classifies questions before processing
- **LLM-based Classification**: Uses GPT to determine "summery" or "qna" classification
- **Clear Routing**: Shows classification step and agent selection process
- **Validation**: Ensures classification results are within expected options

### Common Features

- **Simple API**: Clean functional interface with pattern selection
- **Multiple Agents**: Choose between Timeline, QnA, or Combined agents
- **Verbose Logging**: Track processing steps with detailed prints

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Command Line

```bash
# Interactive agent selector
python main.py

# Run QnA agent with interactive mode
python qna_main.py

# Run Combined agent with intelligent tool selection
python combined_main.py

# Run Classifier agent with explicit classification
python classifier_main.py
```

### Agent Options

When running `python main.py`, you'll see:

```
🚀 Insurance AI Agents
==================================================
1. Timeline Agent - Creates chronological timelines
2. QnA Agent - Answers questions using RAG
3. Combined Agent - Intelligently chooses the right tool
4. Classifier Agent - Pre-classifies questions then routes to agents
5. Run all agents separately
==================================================
Select an agent (1/2/3/4/5):
```

### QnA Agent Features

The QnA agent includes:

- **Automatic Vector Store Creation**: Builds knowledge base from events.txt
- **Semantic Search**: Finds 3 most relevant chunks for each question
- **Sample Questions**: Pre-loaded test questions for demonstration
- **Interactive Mode**: Ask custom questions in real-time

### Example QnA Session

```
❓ Your question: When did John Smith purchase his auto insurance?
💡 Answer: John Smith purchased auto insurance on January 15, 2023, with a premium of $1,200 annually.

❓ Your question: What was the settlement amount for the claim?
💡 Answer: The settlement check was $2,350 to cover the vehicle repairs.
```

### Combined Agent Intelligence

The Combined Agent automatically selects the right tool:

```
❓ Question: "Create a timeline of events"
🧠 Agent chooses: Timeline Tool
📅 Result: Chronological timeline generated

❓ Question: "What was the claim number?"
🧠 Agent chooses: QnA Tool
💡 Result: CL-2023-0045

❓ Question: "Show me a summary timeline"
🧠 Agent chooses: Timeline Tool
📅 Result: Organized timeline summary

❓ Question: "When was the adjuster assigned?"
🧠 Agent chooses: QnA Tool
💡 Result: March 15, 2023
```

### Classifier Agent Flow

The Classifier Agent uses explicit classification before routing:

```
❓ Question: "Create a timeline of insurance events"
🔍 Step 1: Classifying question...
✅ Classification result: summery
🕐 Step 2: Routing to Timeline Agent...
📅 Result: Chronological timeline generated

❓ Question: "What was the settlement amount?"
🔍 Step 1: Classifying question...
✅ Classification result: qna
🤖 Step 2: Routing to QnA Agent...
💡 Result: $2,350
```

## Architecture

### Timeline Agent Processing Patterns:

- **Map-Reduce**: Splits text into chunks, extracts events from each chunk (map), then combines all events into final timeline (reduce)
- **Refine**: Processes first chunk to create initial timeline, then iteratively refines with each subsequent chunk

### QnA Agent RAG Pipeline:

1. **Document Loading**: Reads events.txt content
2. **Text Splitting**: Creates 200-character chunks with 30-character overlap
3. **Embedding**: Uses OpenAI embeddings to create vector representations
4. **Vector Storage**: Stores embeddings in Chroma database
5. **Query Processing**: Embeds user question and retrieves 3 most relevant chunks
6. **Answer Generation**: Sends context and question to LLM for answer generation

### Combined Agent Intelligence:

The Combined Agent uses both pipelines seamlessly:

- **Timeline Patterns**: For requests containing "timeline", "chronological", "summary", "events"
- **QnA Pipeline**: For specific questions like "when", "what", "how much", "where"
- **Automatic Detection**: ReAct pattern analyzes question intent and selects appropriate tool

### Classifier Agent Architecture:

The Classifier Agent uses explicit pre-classification:

1. **Question Input**: User provides a question
2. **LLM Classification**: Sends question + options ["summery", "qna"] to GPT-4o-mini
3. **Classification Result**: Returns exactly one option without extra text
4. **Agent Routing**: Routes to Timeline Agent (summery) or QnA Agent (qna)
5. **Validation**: Ensures classification is within expected options, fallback if invalid

### Agent Framework:

- Uses LangChain's ReAct agent pattern for all agents
- Powered by GPT-4o-mini for optimal performance
- Tools are dynamically created based on selected agent type
- Comprehensive logging for debugging and monitoring
- Combined agent has access to both tool sets for intelligent selection

## Example Output

```
2023-01-15 - POLICY_START - Auto insurance purchased
2023-02-01 - PAYMENT_MADE - First premium payment made
2023-03-11 - INCIDENT_REPORTED - Incident reported to insurance company
2023-03-12 - CLAIM_FILED - Claim officially filed with claim number CL-2023-0045
2023-03-15 - ADJUSTER_ASSIGNED - Adjuster assigned to the case
2023-03-16 - INVESTIGATION_STARTED - Investigation began to determine fault and assess damages
2023-04-02 - INVESTIGATION_COMPLETED - Investigation completed; John not at fault
2023-04-05 - CLAIM_APPROVED - Claim approved; settlement check issued
2024-01-15 - POLICY_RENEWAL - Policy automatically renewed with premium increase
```

## Project Structure

```
summerize-agent/
├── main.py                 # Entry point with agent selection
├── qna_main.py            # QnA agent with interactive mode
├── combined_main.py       # Combined agent with intelligent tool selection
├── classifier_main.py     # Classifier agent with pre-classification
├── events.txt             # Sample insurance text
├── requirements.txt       # Dependencies (includes RAG libraries)
├── chroma_db/             # Vector database (auto-created)
└── src/
    ├── timeline_tool.py   # Map-reduce and refine implementations
    ├── qna_tool.py        # RAG pipeline implementation
    ├── classifier.py      # Question classification functionality
    └── prompts.py         # Specialized prompts for all agents
```

## Key Features

- **Multi-Agent System**: Choose between Timeline, QnA, Combined, or Classifier agents
- **Intelligent Tool Selection**: Combined agent automatically chooses the right tool for each question
- **Explicit Classification**: Classifier agent pre-classifies questions before routing
- **RAG Implementation**: Complete RAG pipeline with vector storage and retrieval
- **Interactive Experience**: Real-time question answering with user-friendly interface
- **Simple & Clean**: Minimal code using LangChain's proven patterns
- **Functional Approach**: Clean functions without complex class hierarchies
- **Comprehensive Logging**: Detailed prints to track RAG pipeline execution
- **Persistent Storage**: Vector database persists between sessions for efficiency
- **Efficient**: Optimized chunking for both timeline and QnA processing

## Dependencies

The project uses modern LangChain components:

- `langchain-openai`: OpenAI models and embeddings
- `langchain-chroma`: Vector database integration
- `langchain-text-splitters`: Advanced text chunking
- `chromadb`: High-performance vector database
- `sentence-transformers`: Additional embedding support
