from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import START, END, StateGraph

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

def decide_to_generate(state: GraphState):
    print("---ASSESS GRADED DOCUMENTS---")

    if state['web_search']:
        print('---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION')
        return WEBSEARCH
    else:
        print('---DECISION: GENERATE---')
        return GENERATE

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.add_edge(START, RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH:WEBSEARCH, GENERATE:GENERATE})
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path='graph.png')