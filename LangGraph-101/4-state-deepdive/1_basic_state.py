from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class SimpleState(TypedDict):
    count: int

def increment(state:SimpleState)->SimpleState:
    return {"count": state["count"] + 1}

def should_continue(state:SimpleState):
    if state['count'] < 5:
        return "continue"
    else:
        return "stop"

state_graph = StateGraph(SimpleState)

state_graph.add_node("increment", increment)

state_graph.add_edge(START, "increment")
state_graph.add_conditional_edges("increment", should_continue, {"continue":"increment", "stop":END})

app = state_graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')

state = {"count":0}
result = app.invoke(state)

print(result)
