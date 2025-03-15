import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    task='text-generation',
    max_new_tokens=512,
    do_sample=False)

chat_model = ChatHuggingFace(llm=llm)

system_message = SystemMessage(content="You are a helpful assistant")
chat_history = MongoDBChatMessageHistory(session_id='mistral_session1',
                                         connection_string=os.getenv("MONGO_URI"),
                                        database_name="mistral_history",
                                        collection_name='chathistory')

messages = [system_message] + chat_history.messages
while True:
    prompt = input("Enter Prompt: ")
    if prompt == 'quit':
        break
    human_message = HumanMessage(content=prompt)
    messages.append(human_message)

    res = chat_model.invoke(messages)
    response = res.content
    ai_message = AIMessage(content=response)
    messages.append(ai_message)
    print(f"\nAI :{response}\n")

    chat_history.add_user_message(human_message)
    chat_history.add_ai_message(ai_message)

print(chat_history.messages)