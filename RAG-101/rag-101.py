# Import required libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os

# Define the LLM model to be used
LLM_MODEL = "deepseek-r1:1.5b"
BASE_URL = "http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
COLLECTION_NAME = "rag_collection_demo_1"

# Configure ChromaDB
def configure_chromadb():
    return chromadb.PersistentClient(path=os.path.join(os.getcwd(), "vectordb"))

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

# Initialize the embedding function with Ollama embeddings
def initialize_embedding_function():
    return ChromaDBEmbeddingFunction(
        OllamaEmbeddings(
            model=LLM_MODEL,
            base_url=BASE_URL
        )
    )

# Define a collection for the RAG workflow
def get_or_create_collection(chroma_client, embedding_function):
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "A collection for RAG with Ollama - Demo1"},
        embedding_function=embedding_function
    )

# Function to add documents to the ChromaDB collection
def add_documents_to_collection(collection, documents, ids):
    collection.add(documents=documents, ids=ids)

# Function to query the ChromaDB collection
def query_chromadb(collection, query_text, n_results=1):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results["documents"], results["metadatas"]

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    llm = OllamaLLM(model=LLM_MODEL)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(collection, query_text):
    retrieved_docs, metadata = query_chromadb(collection, query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print("######## Augmented Prompt ########")
    print(augmented_prompt)
    response = query_ollama(augmented_prompt)
    return response

def main():
    chroma_client = configure_chromadb()
    embedding_function = initialize_embedding_function()
    collection = get_or_create_collection(chroma_client, embedding_function)

    # Example: Add sample documents to the collection
    documents = [
        "Artificial intelligence is the simulation of human intelligence processes by machines.",
        "Python is a programming language that lets you work quickly and integrate systems more effectively.",
        "ChromaDB is a vector database designed for AI applications."
    ]
    doc_ids = ["doc1", "doc2", "doc3"]
    add_documents_to_collection(collection, documents, doc_ids)

    # Example usage
    query = "What is artificial intelligence?"  # Change the query as needed
    response = rag_pipeline(collection, query)
    print("######## Response from LLM ########\n", response)

if __name__ == "__main__":
    main()