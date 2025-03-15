from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema import StrOutputParser

# Given a product, give pros and cons for the product's features
# get features for the product
# get cons of the features
# get pros of the features
# combine and return output

chat_model = ChatOllama(model="llama2:latest", temperature=1.0)

messages = [
    ("system", "You are an expert product reviewer."),
    ("human", "List the main features of the product {product_name}.")]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)

def give_pros(features):
    messages = [
        ("system", "You are a helpful reviewer who gives pros of a product given its features"),
        ("human", "Given these features : {features}, list pros of the features")
    ]
    pros_template = ChatPromptTemplate.from_messages(messages=messages)
    return pros_template.format_prompt(features=features)

def give_cons(features):
    messages = [
        ("system", "You are a helpful reviewer who gives cons of a product given its features"),
        ("human", "Given these features : {features}, list cons of the features")
    ]
    cons_template = ChatPromptTemplate.from_messages(messages=messages)
    return cons_template.format_prompt(features=features)

def combine_pros_cons(pros, cons):
    return f"""Here are the pros : \n{pros}\n\n Here are the cons : \n{cons}\n\n"""

pros_branch_chain = RunnableLambda(lambda x: give_pros(x)) | chat_model | StrOutputParser()
cons_branch_chain = RunnableLambda(lambda x: give_cons(x)) | chat_model | StrOutputParser()

chain = (
    prompt_template
    | chat_model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x['branches']['pros'], x['branches']['cons']))
    )

result = chain.invoke({"product_name":"Apple Macbook Pro"})

print(result)
