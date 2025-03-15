from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOllama(model='llama2:latest', temperature=0)
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

db = Chroma(collection_name='0-rag', persist_directory='./vector_stores/0-rag', embedding_function=embeddings)

print("Number of documents stored:", db._collection.count())

retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.1})

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is.")

messages = [
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder('chat_history'), #variable name
    ("human", "{input}")]

contextualize_q_prompt = ChatPromptTemplate.from_messages(messages=messages)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}")

messages = [
    ("system", qa_system_prompt),
    MessagesPlaceholder('chat_history'),
    ("human", "{input}")]

qa_prompt = ChatPromptTemplate.from_messages(messages=messages)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
history_aware_retriever =  create_history_aware_retriever(llm, retriever, contextualize_q_prompt)   # This uses the LLM to help reformulate the question based on chat history
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

print("Start chatting with the AI! Type 'exit' to end the conversation.")
chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})

    print(f"AI: {result['answer']}")

    # Update the chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result["answer"]))
