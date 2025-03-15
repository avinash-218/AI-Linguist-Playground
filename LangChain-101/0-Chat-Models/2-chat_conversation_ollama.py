from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOllama(
    model="deepseek-r1:14b",
    temperature=0)

system_message = SystemMessage(content="You are a helpful assistant")
chat_history = [system_message]

while True:
    prompt = input("Enter Prompt: ")
    if prompt == 'quit':
        break
    chat_history.append(HumanMessage(content=prompt))

    res = chat_model.invoke(chat_history)
    response = res.content
    chat_history.append(AIMessage(content=response))
    print(f"\nAI :{response}\n")

print(chat_history)