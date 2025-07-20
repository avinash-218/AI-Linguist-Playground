from dotenv import load_dotenv
import operator
from typing import  Annotated, Any, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

load_dotenv()

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    which: str

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state:State)-> Any:  # make the oject to be used as a function.
        import time
        time.sleep(1)
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}


def route_bc_or_cd(state:State)-> Sequence[str]:
    if state['which'] == 'cd':
        print('going to cd')
        return ['c', 'd']
    print('going to bc')
    return ['b', 'c']

intermediates = ['b', 'c', 'd']

builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))
builder.add_node("e", ReturnNodeValue("I'm E"))

builder.add_edge(START, 'a')
builder.add_conditional_edges('a', route_bc_or_cd, intermediates)   # intermediates specify what are the possible nodes a can go

for node in intermediates:
    builder.add_edge(node, 'e')
builder.add_edge('e', END)

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

if __name__=='__main__':
    # graph.invoke({"aggregator":[], 'which':''}, config={"configurable":{"thread_id":'1'}})
    graph.invoke({"aggregator":[], 'which':'cd'}, config={"configurable":{"thread_id":'1'}})