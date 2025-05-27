from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import tool
import datetime

@tool
def get_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    cur_time = datetime.datetime.now()
    formatted_time = cur_time.strftime(format)
    return formatted_time

tools = [get_time]
    
llm = ChatOllama(model='llama3:8b')

query = 'What is the current time in London (You are in India)? Just show me the current time and not the date'

# prompt_template = hub.pull('hwchase17/react')

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt_template = PromptTemplate(input_variables=["tools", "tool_names", "input", "agent_scratchpad"], template=template)

filled_prompt = prompt_template.format(
    tools=", ".join([tool.name for tool in tools]),
    tool_names=", ".join([tool.name for tool in tools]),
    input=query,
    agent_scratchpad=""
)
print(filled_prompt)

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

agent_executor.invoke({"input":query})