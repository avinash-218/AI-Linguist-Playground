from langgraph.graph import START, END, StateGraph, add_messages
from typing import TypedDict, Annotated, List
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import uuid

llm = ChatGroq(model="llama-3.1-8b-instant")

class State(TypedDict):
    linkedin_topic: str
    generated_post: Annotated[List[str], add_messages]  # list of generated linkedin post
    human_feedback: Annotated[List[str], add_messages]  # list of human feedbacks

def model(state: State):
    print("[model_node] Generating content")
    linkedin_topic = state['linkedin_topic']
    feedback = state['human_feedback'] if 'human_feedback' in state else ["No feedback yet"]
    generated_post = state['generated_post'] if 'generated_post' in state else []

    prompt = f"""
        LinkedIn Topic: {linkedin_topic}
        Previous Post: {generated_post[-1] if generated_post else ""}
        Human Feedback: {feedback[-1] if feedback else "No feedback yet"}

        Improve or regenerate the LinkedIn post based on the topic, prior content, and feedback.
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert LinkedIn content writer"),
        HumanMessage(content=prompt)])
    
    generated_linkedin_post = response.content

    print(f"[model_node] Generated content: {generated_linkedin_post}")

    return {"generated_post": [AIMessage(content=generated_linkedin_post)]}

def human_node(state: State):
    print("[human_node] Awaiting Human feedback")

    generated_post = state['generated_post']

    user_feedback = interrupt({"generated_post": generated_post, "message": "Provide feedback or type 'done' to finish"})

    print(f"[human_node] Received Human feedback: {user_feedback}")

    if user_feedback.lower() == "done":
        return Command(update={"human_feedback": state['human_feedback'] + ['Finalised']}, goto='end_node')

    return Command(update={"human_feedback": state['human_feedback'] + [user_feedback]}, goto='model')

def end_node(state:State):
    print("\n[end_node] Process Finished")
    print("Final Generated Post:", state['generated_post'][-1].content)
    print("Human Feedbacks:", state['human_feedback'])
    return state # or just return

graph = StateGraph(State)

graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")

graph.add_edge("end_node", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
config = {"configurable":{"thread_id": uuid.uuid4()}}

linkedin_topic = input("Enter the LinkedIn Topic: ")
initial_state = {"linkedin_topic": linkedin_topic, "generated_post": [], "human_feedback": []}

events = app.stream(initial_state, config=config)

for event in events:
    for node_id, value in event.items(): # if interrupt isreached continuously ask for feedback

        if node_id == '__interrupt__':
            while True:
                user_feedback = input("Provide feedback (or type 'done' when finished): ")

                app.invoke(Command(resume=user_feedback), config=config)

                if user_feedback.lower() == 'done':
                    break
