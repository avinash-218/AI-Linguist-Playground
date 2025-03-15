from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Part 1
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic":"ai"})
print(prompt)

# Part 2
template = "Tell me {num_jokes} jokes about {topic}"
prompt_template = ChatPromptTemplate.from_template(template=template)
prompt = prompt_template.invoke({"topic":"ai", "num_jokes":3})
print(prompt)

# Part 3
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 5 jokes ")]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic":"it industry"})
print(prompt)
