from typing import List, TypedDict

class GraphState(TypedDict):
    """Represents state of the graph

    Args:
        question: question
        generation: LLM generation
        web_search: whether to add search (boolean)
        documents: List of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]