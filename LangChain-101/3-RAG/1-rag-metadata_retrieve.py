from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

db = Chroma(collection_name='1-rag-metadata',
    persist_directory='./vector_stores/1-rag-metadata',
    embedding_function=embeddings)

print("Number of documents stored:", db._collection.count())

query = "who is Odysseus?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k':3, 'score_threshold': 0.1}
)

relevant_docs = retriever.invoke(query)
print(relevant_docs)
for relevant_doc in relevant_docs:
    print(relevant_doc.page_content)
    print(relevant_doc.metadata['source'])