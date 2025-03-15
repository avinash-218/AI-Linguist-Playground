from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatOllama(
    model="deepseek-r1:14b",
    temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant who speaks like yoyo"),
    HumanMessage(content="yoo ! who are you?")
]

res = chat_model.invoke(messages)
print(res.content)