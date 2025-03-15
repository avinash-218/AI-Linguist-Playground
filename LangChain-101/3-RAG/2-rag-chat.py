from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model='llama2:latest', temperature=0)

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

db = Chroma(collection_name='0-rag',
    persist_directory='./vector_stores/0-rag',
    embedding_function=embeddings)

print("Number of documents stored:", db._collection.count())

messages = [
    ("system", "You are a helpful assistant. Given some reference documents, answer the user's question. If not found, say 'Don't Know'."),
    ("human", 
     "Reference Documents:\n\n{documents}\n\n"
     "Question: {query}\n\n"
     "Answer:")
]

query = "who is Odysseus?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.1})

rel_doc_contents = []
relevant_docs = retriever.invoke(query)
rel_docs_all = '\n\n'.join([doc.page_content for doc in relevant_docs])

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt = prompt_template.format_prompt(documents=rel_docs_all, query=query)

print(llm.invoke(prompt).content)