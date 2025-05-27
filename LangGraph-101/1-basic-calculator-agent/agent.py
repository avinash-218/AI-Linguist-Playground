import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from agent_tools import multiply, divide, add, subtract

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

tools = [multiply, divide, add, subtract]

llm_with_tools = model.bind_tools(tools=tools)

sys_message = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def arithmetic_calculator(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_message] + state['messages'])]}

# Graph
builder = StateGraph(MessagesState)

builder.add_node("assistant", arithmetic_calculator)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")  # a + b * c

app = builder.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')

prompt = input("Enter the arithmetic problem to solve : ")
messages = [HumanMessage(content=prompt)]

messages = app.invoke({"messages": messages})

for m in messages['messages']:
    m.pretty_print()
