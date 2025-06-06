from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama.chat_models import ChatOllama

class GradeAnswer(BaseModel):
    binary_score: bool = Field(description="Answer addresses the question, 'yes' or 'no'")

llm = ChatOllama(model='llama3:8b', temperature=0.2)

structured_llm  = llm.with_structured_output(GradeAnswer)

system = """
You are a grader assessing whether an answer addresses / resolves a question.\n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm

