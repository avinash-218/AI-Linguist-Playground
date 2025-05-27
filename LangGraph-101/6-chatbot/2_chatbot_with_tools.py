from typing import Annotated, TypedDict
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class ChatBotWithTools(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: ChatBotWithTools):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def tools_router(state: ChatBotWithTools):
    last_message = state['messages'][-1]

    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

tool_node = ToolNode(tools=tools)

graph = StateGraph(ChatBotWithTools)
graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

# app.get_graph().draw_mermaid_png(output_file_path="graph.png")
print(app.get_graph().draw_ascii())
print()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    else:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})

        print("AI:", result['messages'][-1].content)