from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model='llama3:8b', temperature=0.2)
prompt = hub.pull('rlm/rag-prompt')

generation_chain= prompt | llm | StrOutputParser()

