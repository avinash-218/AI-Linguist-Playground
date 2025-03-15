import os
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'books/odyssey.txt')
text_loader = TextLoader(file_path=file_path, encoding='utf-8')
documents = text_loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents=documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

db = Chroma.from_documents(collection_name='0-rag',
                           persist_directory='./vector_stores/0-rag',
                           documents=docs,
                           embedding=embeddings)

print("Number of documents stored:", db._collection.count())