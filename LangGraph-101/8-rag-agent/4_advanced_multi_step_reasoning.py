from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from typing import TypedDict, List
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

embedding_function = OllamaEmbeddings(model="llama3-groq-tool-use")

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
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

llm = ChatOllama(model="llama3:8b")

template = """Answer the question based on the following context and the Chathistory. Especially take the latest question into consideration:

Chathistory: {history}

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm

class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage

class GradeQuestion(BaseModel):
    relevant: str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'. Expected answer is only Yes or No"
    )

def question_rewriter(state: AgentState):
    print(f"Entering question rewriter")

    # Reset State Variable
    state["documents"] = []
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:  # rephrase actually
        conversation = state["messages"][:-1]
        current_question = state["question"].content

        messages = [
            SystemMessage(
                content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."
            )
        ]
        messages.extend(conversation)  # past messages
        messages.append(HumanMessage(content=current_question))  # question

        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        prompt = rephrase_prompt.format()
        response = llm.invoke(prompt)
        state["rephrased_question"] = response.content.strip()
        print(f"question_rewriter: Rephrased Question: {state['rephrased_question']}")

    else:
        state["rephrased_question"] = state["question"].content

    return state

def question_classifier(state: AgentState):
    print(f"Entering question_classifier")
    system_message = SystemMessage(content=""" You are a classifier that determines whether a user's question is about one of the following topics 
    
    1. Gym History & Founder
    2. Operating Hours
    3. Membership Plans 
    4. Fitness Classes
    5. Personal Trainers
    6. Facilities & Equipment
    
    If the question Is about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'.
    """)
    human_message = HumanMessage(
        content=f"User question: {state['rephrased_question']}"
    )
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    res = grader_llm.invoke({})
    state["on_topic"] = res.relevant.strip()
    print(f"question_classifier: on_topic: {state['on_topic']}")
    return state

def on_off_topic_router(state: AgentState):
    print(f"Entering on_off_topic_router")
    on_topic = state.get("on_topic", "").strip().lower()
    if on_topic == "yes":
        print("Routing to retriever")
        return "retrieve"
    else:
        print("Routing to off_topic response")
        return "off_topic_response"

def retrieve(state: AgentState):
    print(f"Entering retrieve")
    docs = retriever.invoke(state["rephrased_question"])
    print(f"Number of documents retrieved: {len(docs)}")
    state["documents"] = docs
    return state

class GradeDocument(BaseModel):
    relevant: str = Field(
        description="Document is relevant to the question? If yes -> 'Yes' if not -> 'No'. Expected answer is only Yes or No"
    )

def retrieval_grader(state: AgentState):
    print("Entering retrieval_grader")
    system_message = SystemMessage(
        content="""You are a grader assessing the relevance of a retrieved document to a user question.
        Only answer with 'Yes' or 'No'.

        If the document contains information relevant to the user's question, respond with 'Yes'.
        Otherwise, respond with 'No'."""
    )
    structured_llm = llm.with_structured_output(GradeDocument)

    relevant_docs = []
    for doc in state["documents"]:
        human_message = HumanMessage(
            content=f"User question: {state['rephrased_question']}\n\n Retrieved Document: {doc.page_content}"
        )
        grader_prompt = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        )
        grader_llm = grader_prompt | structured_llm
        result = grader_llm.invoke()
        print(
            f"Grader document: {doc.page_content[:30]}... Result: {result.relevant.strip()}"
        )

        if result.relevant.strip().lower() == "yes":
            relevant_docs.append(doc)

    state["documents"] = relevant_docs
    state["proceed_to_generate"] = True
    print(f"retrieval_grader: proceed_to_generate: {state['proceed_to_generate']}")
    return state

def proceed_router(state: AgentState):
    print(f"Entering proceed_router")
    rephrase_count = state.get("rephrase_count", 0)
    if state.get("proceed_to_generate", False):
        print("Routing to generate_answer")
        return "generate_answer"
    elif rephrase_count >= 2:
        print("Maximum rephrase count reached. Cannot find relevant documents")
        return "cannot_answer"
    else:
        print("Routing to question_rewriter")
        return "refine_question"

def refine_question(state: AgentState):
    print(f"Entering refine_question")
    rephrased_count = state.get("rephrase_count", 0)
    question_to_refine = state["rephrased_question"]

    system_message = SystemMessage(
        content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results. Provide a slightly adjusted version of the question."""
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
    )

    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    prompt = refine_prompt.format()
    response = llm.invoke(prompt)
    state["rephrased_question"] = response.content.strip()
    state["rephrase_count"] = rephrased_count + 1
    print(f"refine_question: Refined Question: {state['rephrased_question']}")
    return state

def generate_answer(state: AgentState):
    print(f"Entering generate_answer")
    history = state["messages"]
    docs = state["documents"]
    rephrased_question = state["rephrased_question"]

    context_text = "\n\n".join([doc.page_content for doc in docs])

    response = rag_chain.invoke(
        {"context": context_text, "question": rephrased_question, "history": history}
    )

    generation = response.content.strip()

    state["messages"].append(AIMessage(content=generation))
    print(f"generate_answer: Generation: {generation}")
    return state

def cannot_answer(state: AgentState):
    print("Entering cannot_answer")
    state["messages"].append(
        AIMessage(
            content="I'm sorry, I can't find the information yor are looking for."
        )
    )
    return state

def off_topic_response(state: AgentState):
    print("Entering off_topic_response")
    state["messages"].append(
        AIMessage(content="I'm sorry, I can't answer this question.")
    )
    return state

checkpointer = MemorySaver()

graph = StateGraph(AgentState)

graph.add_node("question_rewriter", question_rewriter)
graph.add_node("question_classifier", question_classifier)
graph.add_node("retrieve", retrieve)
graph.add_node("retrieval_grader", retrieval_grader)
graph.add_node("refine_question", refine_question)
graph.add_node("generate_answer", generate_answer)
graph.add_node("cannot_answer", cannot_answer)
graph.add_node("off_topic_response", off_topic_response)

graph.add_edge(START, "question_rewriter")
graph.add_edge("question_rewriter", "question_classifier")
graph.add_conditional_edges(
    "question_classifier",
    on_off_topic_router,
    {
        "on_topic": "retrieve",
        "off_topic_response": "off_topic_response",
    },
)

graph.add_edge("off_topic_response", END)

graph.add_edge("retrieve", "retrieval_grader")
graph.add_conditional_edges(
    "retrieval_grader",
    proceed_router,
    {
        "can_answer": "generate_answer",
        "cant_answer": "cannot_answer",
        "refine_again": "refine_question",
    },
)
graph.add_edge("generate_answer", END)
graph.add_edge("cannot_answer", END)
graph.add_edge("refine_question", "retrieve")

app = graph.compile(checkpointer=checkpointer)

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

## Off topic question
# print('-'*100)
# query = "What does the company Apple do?"
# print(f"Human: {query}")
# input_data = {"question": HumanMessage(content=query)}
# res = app.invoke(input=input_data, config={"configurable": {"thread_id": 1}})
# print(f"AI: {res['messages'][-1].content}")

########################################

# ## No docs found
# print('-'*100)
# query = "What is the cancelation policy for Peak Performance Gym memberships?"
# print(f"Human: {query}")
# input_data = {
#     "question": HumanMessage(
#         content=query
#     )
# }
# res = app.invoke(input=input_data, config={"configurable": {"thread_id": 2}})
# print(f"AI: {res['messages'][-1].content}")

########################################

## Rag with History
print('-'*100)
query = "Who founded Peak Performance Gym?"
print(f"Human: {query}")
input_data = {"question": HumanMessage(content=query)}
res = app.invoke(input=input_data, config={"configurable": {"thread_id": 3}})
print(f"AI: {res['messages'][-1].content}")

# print('-'*100)
# query = "When did he start it?"
# print(f"Human: {query}")
# input_data = {"question": HumanMessage(content=query)}
# res = app.invoke(input=input_data, config={"configurable": {"thread_id": 4}})
# print(f"AI: {res['messages'][-1].content}")
# print('-'*100)
