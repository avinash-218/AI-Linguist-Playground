import os
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

current_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(current_dir, 'books')

docs = []
for file in os.listdir(dir_path):
    file_path = os.path.join(current_dir, f'books/{file}')
    text_loader = TextLoader(file_path=file_path, encoding='utf-8')
    documents = text_loader.load()
    for doc in documents:
        doc.metadata = {"source":file}
        docs.append(doc)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents=documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

db = Chroma.from_documents(collection_name='1-rag-metadata',
                           persist_directory='./vector_stores/1-rag-metadata',
                           documents=docs,
                           embedding=embeddings)

print("Number of documents stored:", db._collection.count())