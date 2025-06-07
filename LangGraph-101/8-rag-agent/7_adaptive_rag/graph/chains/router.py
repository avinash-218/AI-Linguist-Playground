from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""
    datasource : Literal['vectorstore', 'websearch'] = Field(default=..., description="Given a user question, choose to route it to web search or vector store.") 

llm = ChatOllama(model='llama3:8b', temperature=0.2)
structured_llm  = llm.with_structured_output(RouteQuery)

system = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web search."""

route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

question_router = route_prompt | structured_llm