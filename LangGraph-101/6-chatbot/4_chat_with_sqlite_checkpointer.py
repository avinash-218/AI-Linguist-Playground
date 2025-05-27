from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

load_dotenv()

sqlite_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)
config = {"configurable": {"thread_id": 1}}

llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    else:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        print("AI:", result['messages'][-1].content)
