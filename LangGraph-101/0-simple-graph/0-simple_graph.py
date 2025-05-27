from typing_extensions import TypedDict
import random
from typing import Literal
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    graph_info : str

def start_play(state : State):
    print("Start Play node has been called")
    return {"graph_info": state['graph_info'] + "I am planning to play"}

def cricket(state:State):
    print("Cricket node has been called")
    return {"graph_info": state['graph_info'] + " Cricket"}

def badminton(state:State):
    print("Badminton node has been called")
    return {"graph_info": state['graph_info'] + " Badminton"}

def random_play(state:State)-> Literal['cricket','badminton']:
    if random.random()>0.5:
        return "cricket"
    else:
        return "badminton"
    
# build grpah
graph = StateGraph(State)

#add nodes
graph.add_node("start_play", start_play) 
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)

#add edges
graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play", random_play)
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)

graph_builder = graph.compile()

graph_builder.get_graph().draw_mermaid_png(output_file_path='graph.png')

res = graph_builder.invoke({"graph_info": "My name is Avinash."})

print(res)