import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOllama(
    model="deepseek-r1:14b",
    temperature=0)

system_message = SystemMessage(content="You are a helpful assistant")
chat_history = MongoDBChatMessageHistory(session_id='deepseek_session1',
                                         connection_string=os.getenv("MONGO_URI"),
                                        database_name="deepseek_history",
                                        collection_name='chathistory2')

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