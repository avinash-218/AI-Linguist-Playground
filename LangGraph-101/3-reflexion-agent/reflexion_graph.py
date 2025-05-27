from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import START, END, MessageGraph
from chains import first_response, revise_response
from execute_tools import execute_tool

graph = MessageGraph()
MAX_ITERATIONS = 2

graph.add_node("responder", first_response)
graph.add_node("execute_tool", execute_tool)
graph.add_node("revisor", revise_response)

graph.add_edge(START, "responder")
graph.add_edge("responder", "execute_tool")
graph.add_edge("execute_tool", "revisor")

def event_loop(state: List[BaseMessage]) -> str:
    num_iterations = sum(isinstance(item, ToolMessage) for item in state)
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tool"

graph.add_conditional_edges("revisor", event_loop)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')

response = app.invoke("Write about how small business can leverage AI to grow")

# print(response[-1].tool_calls[0]["args"]["answer"])
print(response, "response")