from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

llm = ChatOllama(model='llama2:latest', temperature=0)
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

db = Chroma(collection_name='0-rag', persist_directory='./vector_stores/0-rag', embedding_function=embeddings)

print("Number of documents stored:", db._collection.count())

retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.1})

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Given some reference documents, answer the user's question. "
               "If the answer is not found in the documents, say 'Don't Know'."),
    ("human", 
     "Reference Documents:\n\n{context}\n\n"
     "Question: {input}\n\n"
     "Answer:")
])

qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain)

query = "who is Odysseus?"
result = rag_chain.invoke({"input": query})

print("\n--- Generated Answer ---")
print(result['answer'])
