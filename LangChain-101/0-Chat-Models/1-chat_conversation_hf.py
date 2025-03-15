from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    task='text-generation',
    max_new_tokens=512,
    do_sample=False)

chat_model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant who speaks like yoyo"),
    HumanMessage(content="yoo ! who are you?")
]

res = chat_model.invoke(messages)
print(res.content)