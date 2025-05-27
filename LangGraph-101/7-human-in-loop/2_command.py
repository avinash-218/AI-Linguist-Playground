from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from langgraph.types import Command

class State(TypedDict):
    text: str

def node_a(state: State):
    print("Node A")
    return Command(goto="node_b", update={"text": state["text"] + "a"})

def node_b(state: State):
    print("Node B")
    return Command(goto="node_c", update={"text": state["text"] + "b"})

def node_c(state: State):
    print("Node C")
    return Command(goto=END, update={"text": state["text"] + "c"})

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)

graph.add_edge(START, "node_a")

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

response = app.invoke({"text": ""})

print(response)