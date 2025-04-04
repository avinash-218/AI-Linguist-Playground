from typing_extensions import Annotated, TypedDict
from typing import Optional
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3:8b")

json_schema = {
    "title": "joke",
    "description": "Joke to tell the user.",
    "type": "object",
    "properties": {
        "setup": {"type": "string", "description": "The setup of the joke"},
        "punchline": {"type": "string", "description": "The punchline of the joke"},
        "rating": {"type": "integer", "description": "How funny the joke is, on a scale of 1 to 10", "default": None,},
    },
    "required": ["setup", "punchline", "rating"],
}

structured_llm = llm.with_structured_output(json_schema)
out = structured_llm.invoke("Tell me a IT joke")
print(out, sep="\n")
