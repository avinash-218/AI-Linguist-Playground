from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, Field

embedding_function = OllamaEmbeddings(model="llama3:8b")

docs = [
    Document(
        page_content="Peak Performance Gym was founded in 2015 by former Olympic athlete Marcus Chen. With over 15 years of experience in professional athletics, Marcus established the gym to provide personalized fitness solutions for people of all levels. The gym spans 10,000 square feet and features state-of-the-art equipment.",
        metadata={"source": "about.txt"},
    ),
    Document(
        page_content="Peak Performance Gym is open Monday through Friday from 5:00 AM to 11:00 PM. On weekends, our hours are 7:00 AM to 9:00 PM. We remain closed on major national holidays. Members with Premium access can enter using their key cards 24/7, including holidays.",
        metadata={"source": "hours.txt"},
    ),
    Document(
        page_content="Our membership plans include: Basic (₹1,500/month) with access to gym floor and basic equipment; Standard (₹2,500/month) adds group classes and locker facilities; Premium (₹4,000/month) includes 24/7 access, personal training sessions, and spa facilities. We offer student and senior citizen discounts of 15% on all plans. Corporate partnerships are available for companies with 10+ employees joining.",
        metadata={"source": "membership.txt"},
    ),
    Document(
        page_content="Group fitness classes at Peak Performance Gym include Yoga (beginner, intermediate, advanced), HIIT, Zumba, Spin Cycling, CrossFit, and Pilates. Beginner classes are held every Monday and Wednesday at 6:00 PM. Intermediate and advanced classes are scheduled throughout the week. The full schedule is available on our mobile app or at the reception desk.",
        metadata={"source": "classes.txt"},
    ),
    Document(
        page_content="Personal trainers at Peak Performance Gym are all certified professionals with minimum 5 years of experience. Each new member receives a complimentary fitness assessment and one free session with a trainer. Our head trainer, Neha Kapoor, specializes in rehabilitation fitness and sports-specific training. Personal training sessions can be booked individually (₹800/session) or in packages of 10 (₹7,000) or 20 (₹13,000).",
        metadata={"source": "trainers.txt"},
    ),
    Document(
        page_content="Peak Performance Gym's facilities include a cardio zone with 30+ machines, strength training area, functional fitness space, dedicated yoga studio, spin class room, swimming pool (25m), sauna and steam rooms, juice bar, and locker rooms with shower facilities. Our equipment is replaced or upgraded every 3 years to ensure members have access to the latest fitness technology.",
        metadata={"source": "facilities.txt"},
    ),
]

db = Chroma.from_documents(docs, embedding_function)

# Retrieve
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

template = """
Answer the question based only on the following context: {context}
Question : {question}
"""

prompt = ChatPromptTemplate.from_template(template)

class GradeQuestion(BaseModel):
    """Boolean value to check whether a question is related to the Peak Performance Gym"""
    score: str = Field(description="Question is about gym? If yes-> 'Yes' if not 'No'")

llm = ChatOllama(model="llama3:8b")
structured_llm = llm.with_structured_output(GradeQuestion)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = prompt | llm

class AgenticState(TypedDict):
    messages: list[BaseMessage]
    documents: list[Document]
    on_topic: str

def question_classifier(state: AgenticState):
    system = """ You are a classifier that determines whether a user's question is about one of the following topics 
    
    1. Gym History & Founder
    2. Operating Hours
    3. Membership Plans 
    4. Fitness Classes
    5. Personal Trainers
    6. Facilities & Equipment
    
    If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'.
    """

    question = state["messages"][-1].content  # get last human message - which is question in this case

    grade_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "User question: {question}")])

    grader_llm = grade_prompt | structured_llm

    res = grader_llm.invoke({"question": question})
    state["on_topic"] = res.score
    return state

def on_off_topic_router(state: AgenticState):
    on_topic = state["on_topic"]
    if on_topic.lower() == "yes":
        return "on_topic"
    else:
        return "off_topic"

def retrieve(state: AgenticState):
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    state["documents"] = docs
    return state

def generate_answer(state: AgenticState):
    docs = state["documents"]
    question = state["messages"][-1].content
    res = rag_chain.invoke({"context": format_docs(docs), "question": question})
    state["messages"].append(AIMessage(content=res.content))
    return state

def off_topic_response(state: AgenticState):
    state["messages"].append(AIMessage(content="I'm sorry, I can't answer this question."))
    return state

graph = StateGraph(AgenticState)

graph.add_node("question_classifier", question_classifier)
graph.add_node("retrieve", retrieve)
graph.add_node("generate_answer", generate_answer)
graph.add_node("off_topic_response", off_topic_response)

graph.add_edge(START, "question_classifier")
graph.add_conditional_edges(
    "question_classifier",
    on_off_topic_router,
    {
        "on_topic": "retrieve",
        "off_topic": "off_topic_response",
    },
)
graph.add_edge("retrieve", "generate_answer")
graph.add_edge("generate_answer", END)
graph.add_edge("off_topic_response", END)

app = graph.compile()

# app.get_graph().draw_mermaid_png(output_file_path="graph.png")
# print(app.get_graph().draw_ascii())

query1 = "Tell me about the Peak Performance Gym membership?"
res = app.invoke(input={"messages": [HumanMessage(content=query1)]})
print('-'*100)
print(f"Human: {query1}\nAI: {res['messages'][-1].content}")
print('-'*100)

query2 = "What does the company apple do?"
res = app.invoke(input={"messages": [HumanMessage(content=query2)]})
print(f"Human: {query2}\nAI: {res['messages'][-1].content}")
print('-'*100)