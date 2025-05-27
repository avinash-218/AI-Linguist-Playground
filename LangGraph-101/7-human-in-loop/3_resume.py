from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

class State(TypedDict):
    text: str

def node_a(state: State):
    print("Node A")
    return Command(goto="node_b", update={"text": state["text"] + "a"})

def node_b(state: State):
    print("Node B")
    human_response = interrupt("Do you want to C or D. Type C/D")
    print(f"Human Response :", human_response)

    if human_response == "C":
        return Command(goto="node_c", update={"text": state["text"] + "b"})
    if human_response == "D":
        return Command(goto="node_d", update={"text": state["text"] + "b"})

def node_c(state: State):
    print("Node C")
    return Command(goto=END, update={"text": state["text"] + "c"})

def node_d(state: State):
    print("Node D")
    return Command(goto=END, update={"text": state["text"] + "d"})

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)

graph.add_edge(START, "node_a")

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

response = app.invoke({"text": ""}, config=config, stream_mode="updates")  # stream mode - output of every nodes

print("\nGraph State Config:", app.get_state(config=config))

print("\nResponse:", response)

print("\nCheckpoint :", app.get_state(config=config).next, sep="")

##### Resume
second_result = app.invoke(Command(resume="D"), config=config, stream_mode="updates")
print("\nResponse:", second_result)
