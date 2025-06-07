from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import START, END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
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

def grade_generation_grounded_in_documents_and_question(state: GraphState):
    print('---CHECK HALLUCINATIONS---')
    question = state['question']
    documents = state['documents']
    generation = state['generation']

    score = hallucination_grader.invoke({"documents":documents, "generation":generation})

    if score.binary_score: # if true
        print('---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---')
        print('---GRADE GENERATION Vs QUESTION---')
        score = answer_grader.invoke({"question":question, "generation":generation})
        if score.binary_score:  # if true
            print('---DECISION: GENERATION ADDRESSES QUESTION---')
            return "useful"
        else:
            print('---DECISION: GENERATION DOES NOT ADDRESS QUESTION---')
            return "not useful"
    else:
        print('---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---')
        return "not supported"

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.add_edge(START, RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH:WEBSEARCH, GENERATE:GENERATE})
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question,{"not supported":GENERATE, "not useful":WEBSEARCH, "useful":END})

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path='graph.png')