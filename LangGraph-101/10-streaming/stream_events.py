import asyncio
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(model='llama-3.1-8b-instant')
llm_with_tools = llm.bind_tools(tools=tools)

def model(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def tools_router(state: AgentState):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END

tool_node = ToolNode(tools=tools)

graph = StateGraph(AgentState)

graph.add_node("model", model)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("model")

graph.add_conditional_edges("model", tools_router)
graph.add_edge("tool_node", "model")

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

# ### stream mode - values
# input = {"messages": ["What's the current weather in Bangalore?"]}
# events = app.stream(input=input, stream_mode="values")
# for event in events: 
#     print(event["messages"])

# ### stream mode - updates
# input = {"messages": ["What's the current weather in Bangalore?"]}
# events = app.stream(input=input, stream_mode="updates")
# for event in events: 
#     print(event)

###
# async def run():
#     input = {"messages": ["Hi, how are you?"]}
#     events = app.astream_events(input=input, version="v2")
#     async for event in events:
#         print(event)
# asyncio.run(run())

###
async def run():
    input = {"messages": ["Hi, how are you?"]}
    events = app.astream_events(input=input, version="v2")
    async for event in events:
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)

asyncio.run(run())