from langchain_ollama import ChatOllama
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub

llm = ChatOllama(model='llama3:8b', temperature=0.1)

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(input: str = None):
    """Returns the current date and time, handling irrelevant inputs."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

tools = [search_tool, get_system_time]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)
