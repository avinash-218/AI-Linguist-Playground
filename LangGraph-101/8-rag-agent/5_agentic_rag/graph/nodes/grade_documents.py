from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState)->GraphState:
    """Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (GraphState): Current state of graph

    Returns:
        GraphState: Filtered out irrelevant documents and updated web_search state
    """
    print('---CHECK DOCUMENT RELEVANCE TO QUESTION---')
    filtered_docs = []
    web_search = False

    for doc in state['documents']:
        score =  retrieval_grader.invoke({
        "document": doc.page_content,
        "question": state['question']})

        grade = score.binary_score
        if grade.lower() == 'yes':
            print('---GRADE: DOCUMENT RELEVANT---')
            filtered_docs.append(doc)
        else:
            print('---GRADE: DOCUMENT NOT RELEVANT---')
            web_search = True

    state['documents'] = filtered_docs
    state['web_search'] = web_search

    return state