from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="llama2:latest", temperature=1.0)

# Part 1
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic":"ai"})
print(chat_model.invoke(prompt).content)
print('-'*100)

# Part 2
template = "Tell me {num_jokes} jokes about {topic}"
prompt_template = ChatPromptTemplate.from_template(template=template)
prompt = prompt_template.invoke({"topic":"ai", "num_jokes":3})
print(chat_model.invoke(prompt).content)
print('-'*100)

# Part 3
messages = [
    ("system", "You are a comedian who tells jokes happening in {topic}."),
    HumanMessage(content="Tell me 5 jokes ")]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic":"it industry"})
print(chat_model.invoke(prompt).content)
print('-'*100)
