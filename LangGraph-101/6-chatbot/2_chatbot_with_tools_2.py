import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# API key setup
os.environ['TAVILY_API_KEY']=os.getenv("TAVILY_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

# tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arxiv papers")

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, description="Query wikipedia")

tavily = TavilySearchResults()

# combine tools
tools = [arxiv, wiki, tavily]

# llm
llm = ChatGroq(model='llama3-8b-8192')

llm_with_tools = llm.bind_tools(tools=tools)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# graph

def tool_calling_llm(state:State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools=tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()

# graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

messages = graph.invoke({"messages":HumanMessage(content="1706.03762")})
messages = graph.invoke({"messages":HumanMessage(content="Tony Stark")})
messages = graph.invoke({"messages":HumanMessage(content="Gold rate at madurai on 23 april 2025")})
messages = graph.invoke({"messages":HumanMessage(content="Hi, My name is Avinash.")})

for m in messages['messages']:
    m.pretty_print()