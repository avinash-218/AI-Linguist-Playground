from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState)-> GraphState:
    print('---WEB SEARCH---')
    question = state['question']
    documents = state['documents']

    tavily_results = web_search_tool.invoke({"query":question})

    joined_tavily_results = '\n'.join(tavily_results)

    web_results = Document(page_content=joined_tavily_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    state['documents'] = documents
    return state