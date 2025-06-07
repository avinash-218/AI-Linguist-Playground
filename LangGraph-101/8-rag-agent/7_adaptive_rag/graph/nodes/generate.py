from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState)-> GraphState:
    print('---GENERATE---')
    question = state['question']
    documents = state['documents']

    state['generation'] = generation_chain.invoke({"context": documents, "question":question})
    return state