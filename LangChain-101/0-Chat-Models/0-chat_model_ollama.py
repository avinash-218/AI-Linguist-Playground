from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatOllama(
    model="deepseek-r1:14b",
    temperature=0)

result = chat_model.invoke("Who are you?")
print(result.content)