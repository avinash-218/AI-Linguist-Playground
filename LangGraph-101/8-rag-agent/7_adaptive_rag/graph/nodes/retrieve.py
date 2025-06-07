from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState)-> GraphState:
    print('---RETRIEVE---')

    documents = retriever.invoke(state['question'])
    return {"documents": documents}

