from fastapi import FastAPI, Query
import requests

app = FastAPI()

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/chat")
async def chat(prompt: str = Query("Hello, Jarvis!", description="Enter a prompt")):
    data = {
        "model": "jarvis",  # Custom model name
        "prompt": prompt,
        "stream": False}
    
    try:
        response = requests.post(OLLAMA_URL, json=data)
        response.raise_for_status()  # Raise error for bad responses
        
        response_json = response.json() # Extract only the "response" field
        return response_json.get("response", "No response received")
    
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
