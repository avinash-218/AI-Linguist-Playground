from fastapi import FastAPI
from pydantic import BaseModel
import requests
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

app = FastAPI()

PERSIST_DIR = r'ChromaDB\4\vector_store'  # Path where your ChromaDB is stored
OLLAMA_MODEL = "deepseek-r1:14b"  # Ollama model to use

embedding = OllamaEmbeddings(model=OLLAMA_MODEL)
vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: list

def retrieve_relevant_docs(query, k=3):
    docs_and_scores = vector_db.similarity_search_with_score(query, k=k)
    relevant_docs = [doc for doc, score in docs_and_scores]
    return relevant_docs

def create_prompt(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
    prompt = f"""
You are an expert assistant for building codes and regulations.
Based on the following context, answer the user's question concisely.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt, sources

@app.post("/chat", response_model=ChatResponse)
def chat_rag(request: ChatRequest):
    query = request.query
    retrieved_docs = retrieve_relevant_docs(query)
    prompt, sources = create_prompt(query, retrieved_docs)

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        llm_response = response.json().get("response", "").strip()
    except Exception as e:
        llm_response = f"Failed to generate response: {str(e)}"
        sources = []

    return ChatResponse(response=llm_response, sources=sources)

@app.get("/")
def root():
    return {"message": "RAG + Ollama Chat API is running 🚀"}
