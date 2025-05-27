from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, add_messages, StateGraph
from langchain_groq import ChatGroq

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant")

GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"

def generate_post(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def get_review_decision(state: State):
    post_content = state["messages"][-1].content

    print("\nCurrent LinkedIn Post:\n")
    print(post_content, "\n")

    decision = input("Post to LinkedIn? (y/n):")

    if decision.lower() in ["yes", "y"]:
        return POST
    else:
        return COLLECT_FEEDBACK

def post(state: State):
    final_post = state["messages"][-1].content
    print("\nFinal LinkedIn Post:\n")
    print(final_post)
    print("\n Post has been approved and is now live on LinkedIn!")

def collect_feedback(state: State):
    feedback = input("AI: How can I improve this post?")
    return {"messages": [HumanMessage(content=feedback)]}

graph = StateGraph(State)

graph.add_node(GENERATE_POST, generate_post)
graph.add_node(GET_REVIEW_DECISION, get_review_decision)
graph.add_node(POST, post)
graph.add_node(COLLECT_FEEDBACK, collect_feedback)

graph.add_edge(START, GENERATE_POST)
graph.add_conditional_edges(GENERATE_POST, get_review_decision)
graph.add_edge(POST, END)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

response = app.invoke({"messages": [HumanMessage(content="Write me a LinkedIn Post on AI Agents taking over content creation")]})

print(response)