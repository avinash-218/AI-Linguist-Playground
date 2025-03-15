from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser

chat_model = ChatOllama(model="llama2:latest", temperature=1.0)

messages = [
    ("system", "You are a comedian who tells jokes happening in {topic}."),
    ("human", "Tell me {num_jokes} jokes ")]

prompt_template = ChatPromptTemplate.from_messages(messages)

chain = prompt_template | chat_model | StrOutputParser()

response = chain.invoke({"topic":"it industry", "num_jokes":5})

print(response)