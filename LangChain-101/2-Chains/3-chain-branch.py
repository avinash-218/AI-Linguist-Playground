# Given a review of a product
# Generate Response based on the sentiment : Positive, Negative, Neutral, Escalate to customer care human agent
# Given a review, classify the sentiment
# if the sentiment is positive, generate a response saying thank you
# if the sentiment is negative, generate a response saying sorry and well fix it
# if the sentiment is neutral, generate a response asking more details
# if unknown sentiment , generate a response to inform that it has been escalated to human agent

from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableBranch

chat_model = ChatOllama(model = "llama2:latest", temperature = 0.8)

messages = [
    ("system", "You are a helpful agent. You can categorize a feedback as postive, negative, neutral based on the emotion"),
    ("human", "Given the feedback of a product, classify the emotion as positive, negative, neutral or escalate : {feedback}")
]

def get_reply_pos(sentiment):
    messages = [
        ("system", "You are a helpful assistant"),
        ("human", "Given the sentiment :{sentiment}, give thankyou note")
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages=messages)
    chain = prompt_template | chat_model | StrOutputParser()
    response = chain.invoke({"sentiment":sentiment})
    return response

def get_reply_neg(sentiment):
    messages = [
        ("system", "You are a helpful assistant"),
        ("human", "Given the sentiment :{sentiment}, give reply to address the negative feedback")
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages=messages)
    chain = prompt_template | chat_model | StrOutputParser()
    response = chain.invoke({"sentiment":sentiment})
    return response


def get_reply_neutral(sentiment):
    messages = [
        ("system", "You are a helpful assistant"),
        ("human", "Given the sentiment :{sentiment}, give reply to ask for more information")
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages=messages)
    chain = prompt_template | chat_model | StrOutputParser()
    response = chain.invoke({"sentiment":sentiment})
    return response

def get_reply_escalation(sentiment):
    messages = [
        ("system", "You are a helpful assistant"),
        ("human", "Given the sentiment :{sentiment}, give reply to inform that it has been escalated to human agent and someone will get in touch with you")
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages=messages)
    chain = prompt_template | chat_model | StrOutputParser()
    response = chain.invoke({"sentiment":sentiment})
    return response

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
branch = RunnableBranch(
    (lambda x: "positive" in x, get_reply_pos),
    (lambda x: "negative" in x, get_reply_neg),
    (lambda x: "neutral" in x, get_reply_neutral),
    get_reply_escalation)

chain = prompt_template | chat_model | StrOutputParser() | branch

response = chain.invoke({"feedback": "Can you tell me more about its features and benefits?"})
print(response)