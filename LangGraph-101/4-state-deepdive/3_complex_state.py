from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
import operator

class SimpleState(TypedDict):
    """Represents the state of the graph.

    Attributes:
        count: The current count. This value is overwritten in each step.
        sum: The running sum of counts. LangGraph automatically updates this
             by adding the new value returned by a node to the existing value.
        history: A list tracking the history of counts. LangGraph automatically
                 updates this by concatenating the new list returned by a node
                 to the existing list.
    """
    count: int
    sum: Annotated[int, operator.add]
    history: Annotated[List[int], operator.concat]

def increment(state: SimpleState) -> SimpleState:
    new_count = state["count"] + 1
    return {
        "count": new_count,
        "sum": new_count,
        "history": [new_count]
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
