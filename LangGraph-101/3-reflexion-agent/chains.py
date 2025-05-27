from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from schema import AnswerQuestion, ReviseAnswer
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="llama3:8b", temperature=0.7)

first_responder_structure = llm.with_structured_output(AnswerQuestion)
revisor_structure = llm.with_structured_output(ReviseAnswer)

# Actor Agent Prompt 
actor_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI researcher. Current time: {time}

        1. {first_instruction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.

        Format your response correctly for the output schema.
        """),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above."),
    ]).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")
first_responder_chain = first_responder_prompt_template | first_responder_structure

# response = first_responder_chain.invoke({"messages": [HumanMessage(content="Write me a blog post on about how small businesses can leverage AI to grow")]})
# print(response)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_prompt_template = actor_prompt_template.partial(first_instruction=revise_instructions)
revisor_chain = revisor_prompt_template | revisor_structure

def first_response(state):
    answer = first_responder_chain.invoke(state)
    return [
        AIMessage(
            content="", 
            tool_calls=[{
                "name": "AnswerQuestion",
                "args": answer.dict(),
                "id": "call_first_response"
            }]
        )
    ]

def revise_response(state):
    revision = revisor_chain.invoke(state)
    return [
        AIMessage(
            content="", 
            tool_calls=[{
                "name": "ReviseAnswer",
                "args": revision.dict(),
                "id": "call_revise_response"
            }]
        )
    ]

