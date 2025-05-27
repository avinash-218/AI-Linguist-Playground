from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, MessageGraph
from chains import generation_chain, reflection_chain

load_dotenv()

graph = MessageGraph()

# Create nodes
REFLECT = "reflect"
GENERATE = "generate"

def generate_node(state):
    return generation_chain.invoke({"messages":state})

def reflect_node(state):
    response = reflection_chain.invoke({"messages":state})
    return [HumanMessage(content=response.content)] # trick as it is response from human

def should_iterate_or_end(state, num_iterations=4):
    if len(state) > num_iterations:
        return END
    return REFLECT

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.add_edge(START, GENERATE)
graph.add_conditional_edges(GENERATE, should_iterate_or_end)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')

response = app.invoke(HumanMessage(content="AI Agents taking over the world"))

print(response)