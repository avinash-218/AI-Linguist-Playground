from typing_extensions import Annotated, TypedDict
from typing import Optional
from langchain_ollama import ChatOllama

llm = ChatOllama(model='llama3:8b')

class Joke(TypedDict):
    """Joke to tell the user"""
    setup: Annotated[str, ..., "The setup of the joke"] # type, required / default value, description
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, on a scale of 1 to 10"]

structured_llm = llm.with_structured_output(Joke)
out = structured_llm.invoke("Tell me a IT joke")
print(out, sep='\n')