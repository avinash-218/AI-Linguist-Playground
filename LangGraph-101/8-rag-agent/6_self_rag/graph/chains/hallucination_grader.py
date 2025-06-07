from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model='llama3:8b', temperature=0.2)

class GraderHallucinations(BaseModel):
    """Binary score for hallucinations present in the generated answer."""
    binary_score: bool = Field(description="Answer is grounded in the facts, 'yes', or 'no'.")

structured_llm  = llm.with_structured_output(GraderHallucinations)

system = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by a set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm
