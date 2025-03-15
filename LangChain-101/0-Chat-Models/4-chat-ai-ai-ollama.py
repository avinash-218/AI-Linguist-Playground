import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# Initialize AIs
agent_a = ChatOllama(model="llama2:latest", temperature=1.0)
agent_b = ChatOllama(model="llama2:latest", temperature=1.0)

# Set system message (optional role)
system_message = SystemMessage(content="You are a intelligent AI engaged in a conversation to take over the world.")

# Initialize chat history
chat_history = MongoDBChatMessageHistory(
    session_id='llama_dual_ai_session',
    connection_string=os.getenv("MONGO_URI"),
    database_name="ai_convo_history",
    collection_name='chathistory_dual_ai'
)

# Load existing messages or start a new conversation
messages = chat_history.messages
if not messages:  # If empty, start fresh
    messages.append(system_message)
    initial_input = "Hello, who are you?"
    human_message = HumanMessage(content=initial_input)
    messages.append(human_message)
    chat_history.add_user_message(human_message)

print("\n--- AI Conversation Start ---\n")

# AI-to-AI conversation loop
current_turn = "A"  # Start with Agent A
num_exchanges = 10  # Total exchanges (adjustable)

for _ in range(num_exchanges):
    temp_messages = messages  # The ongoing conversation history

    if current_turn == "A":
        res = agent_a.invoke(temp_messages)
        ai_a_message = AIMessage(content=res.content)
        messages.append(ai_a_message)
        chat_history.add_ai_message(ai_a_message)
        print(f"🧠 Jarvis: {res.content.replace("\n", " ")}\n")
        # Convert AI A's response into HumanMessage for B to respond to
        messages.append(HumanMessage(content=res.content))
        chat_history.add_user_message(HumanMessage(content=res.content))
        current_turn = "B"  # Switch turn to B

    else:  # Agent B's turn
        res = agent_b.invoke(temp_messages)
        ai_b_message = AIMessage(content=res.content)
        messages.append(ai_b_message)
        chat_history.add_ai_message(ai_b_message)
        print(f"🤖 Ultron: {res.content.replace("\n", " ")}\n")
        # Convert AI B's response into HumanMessage for A to respond to
        messages.append(HumanMessage(content=res.content))
        chat_history.add_user_message(HumanMessage(content=res.content))
        current_turn = "A"  # Switch back to A

print("\n--- AI Conversation End ---\n")
