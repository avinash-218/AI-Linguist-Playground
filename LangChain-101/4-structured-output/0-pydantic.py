from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

llm = ChatOllama(model='llama3:8b')

class Country(BaseModel):
    """Information about a country"""
    name: str = Field(description="name of the country")
    language: str = Field(description="language of the country")
    capital: str = Field(description="capital of the country")

structured_llm = llm.with_structured_output(Country)
out = structured_llm.invoke("Tell me about india")
print(out.name, out.language, out.capital, sep='\n')