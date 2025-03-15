from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema import StrOutputParser

chat_model = ChatOllama(model="llama2:latest", temperature=1.0)

messages = [
    ("system", "You are a comedian who tells jokes happening in {topic}."),
    ("human", "Tell me {num_jokes} jokes ")]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))    #use input dict {"topic":"ai", "num_jokes":3}
invoke_model = RunnableLambda(lambda x: chat_model.invoke(x))
parse_output = RunnableLambda(lambda x: x.content)
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Num Words : {len(x.split())}")

chain = RunnableSequence(first=format_prompt, middle=[invoke_model, parse_output, uppercase_output], last=count_words)
response = chain.invoke({"topic":"ai", "num_jokes":3})

print(response)

# OR
print('-'*100)

chain = prompt_template | invoke_model | StrOutputParser() | uppercase_output | count_words
response = chain.invoke({"topic":"ai", "num_jokes":3})

print(response)