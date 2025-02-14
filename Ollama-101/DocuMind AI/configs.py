from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

PROMPT_TEMPLATE = """You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you dont' know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

PDF_STORAGE_PATH = r'4\document_store\\'
EMBEDDING_MODEL = OllamaEmbeddings(model='deepseek-r1:1.5b')
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model='deepseek-r1:1.5b')