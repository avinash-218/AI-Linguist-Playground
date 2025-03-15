from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    task='text-generation',
    max_new_tokens=512,
    do_sample=False)

chat_model = ChatHuggingFace(llm=llm)

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