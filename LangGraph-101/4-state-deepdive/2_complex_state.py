from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class SimpleState(TypedDict):
    count: int
    sum: int
    history: List[int]

def increment(state: SimpleState) -> SimpleState:
    new_count = state["count"] + 1
    return {
        "count": new_count,
        "sum": state["sum"] + new_count,
        "history": state["history"] + [new_count]
        }

def should_continue(state: SimpleState):
    if state["count"] < 5:
        return "continue"
    else:
        return "stop"

state_graph = StateGraph(SimpleState)

state_graph.add_node("increment", increment)

state_graph.add_edge(START, "increment")
state_graph.add_conditional_edges(
    "increment", should_continue, {"continue": "increment", "stop": END}
)

app = state_graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

state = {"count": 0, "sum": 0, "history": []}
result = app.invoke(state)

print(result)
