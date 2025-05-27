from typing import Annotated, TypedDict
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}

class BasicState(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools=tools)

def model(state: BasicState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def tools_router(state: BasicState):
    last_message = state['messages'][-1]

    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        return "tools"
    else:
        return END

graph = StateGraph(BasicState)
graph.add_node("model", model)
graph.add_node("tools", ToolNode(tools=tools))

graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_router)
graph.add_edge("tools", "model")

app = graph.compile(checkpointer=memory, interrupt_before=['tools'])

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
print()

events = app.stream({"messages": [HumanMessage(content='What is the weather in Chennai right now?')]}, config=config, stream_mode='values')

for event in events:
    event['messages'][-1].pretty_print()

print("\nCheckpoint :", app.get_state(config=config).next, sep="")

# Resume
events = app.stream(None, config=config, stream_mode='values')  #None means: "Don't start a new input; instead, continue from where the last run left off."
for event in events:
    event['messages'][-1].pretty_print()
