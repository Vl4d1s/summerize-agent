import os
from zep_cloud.client import Zep
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from src.prompts import create_qna_prompt
from src.evaluation.faithfulness_evaluation import print_faithfulness_result
from src.evaluation.context_recall_evaluation import print_context_recall_result
from src.evaluation.context_precision_evaluation import print_context_precision_result

def answer_question_with_graph(question: str, with_evaluation: bool = False) -> str:
    """Answer question using Zep Graph pipeline"""
    client = Zep(api_key="z_1dWlkIjoiNzY2M2MyMDItMDA4My00YWM5LTk0ODItZTAxYmY3ODljYzk4In0.4G3YDD0pjJ94x9Ja1gnSB9UHRSz3hxpujGYzPH6a_VE5M4ztKmF93sQkWseIWX8Vhzwv8xvMhGGe0pU48qBmqg")
    user_id = "static-user"
    # Query the graph for relevant facts
    search_results = client.graph.search(
        user_id=user_id,
        query=question,
        scope="edges",
        limit=3
    )
    if search_results.edges:
        context_list = [edge.fact for edge in search_results.edges]
    else:
        context_list = []
    print("Retrieved Context:")    
    for doc in context_list:
        print("--------------------------------")
        print(doc.strip())
    print("--------------------------------")    
    context = "\n\n".join(context_list)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    output_parser = StrOutputParser()
    qna_chain = create_qna_prompt() | llm | output_parser
    answer = qna_chain.invoke({
        "context": context,
        "question": question
    })
    answer = answer.strip()
    if with_evaluation:
        print_context_recall_result(question, answer, context_list)
        print_faithfulness_result(question, answer, context_list)
        print_context_precision_result(question, answer, context_list)
    return answer 