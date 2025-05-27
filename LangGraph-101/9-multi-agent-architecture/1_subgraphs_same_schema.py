from typing import Annotated, TypedDict
from langgraph.graph import START, END, add_messages, StateGraph
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

load_dotenv()

class ChildState(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(model='llama-3.1-8b-instant')

llm_with_tools = llm.bind_tools(tools=tools)

def agent(state: ChildState):
    return {"messages" : [llm_with_tools.invoke(state['messages'])]}

def tools_router(state: ChildState):
    last_message = state['messages'][-1]
    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return END
    
tool_node = ToolNode(tools=tools)

subgraph = StateGraph(ChildState)

subgraph.add_node("agent", agent)
subgraph.add_node("tool_node", tool_node)

subgraph.add_edge(START, "agent")
subgraph.add_conditional_edges("agent", tools_router)
subgraph.add_edge("tool_node", "agent")

search_app = subgraph.compile()

# search_app.get_graph().draw_mermaid_png(output_file_path="child_graph.png")
# res = search_app.invoke({"messages": [HumanMessage(content="How is the weather in Chennai?")]})
# print(res)
# print(res['messages'][-1].content)

# parent graph with same schema
class ParentState(TypedDict):
    messages: Annotated[list, add_messages]

parent_graph = StateGraph(ParentState)

parent_graph.add_node("search_app", search_app)

parent_graph.add_edge(START, "search_app")
parent_graph.add_edge("search_app", END)

parent_app = parent_graph.compile()

# parent_app.get_graph().draw_mermaid_png(output_file_path="parent_graph.png")
res = parent_app.invoke({"messages": [HumanMessage(content="How is the weather in Chennai?")]})
# print(res)
print(res['messages'][-1].content)
