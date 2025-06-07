from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router
from ingestion import retriever

def test_retrieval_grader_answer_yes()->None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke({
        "document": doc_txt,
        "question":question
    })

    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no()->None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke({
        "document": doc_txt,
        "question":"how to make pizza"
    })

    assert res.binary_score == "no"

def test_generation_chain()->None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question":question})
    pprint(generation)

def test_hallucination_grader_answer_yes()->None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question":question})
    res = hallucination_grader.invoke({"documents": docs, "generation": generation})

    assert res.binary_score == True

def test_hallucination_grader_answer_no()->None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res = hallucination_grader.invoke({"documents": docs, "generation": "In order to make dosa we need to start wth batter."})

    assert res.binary_score == False

def test_router_to_vectorstore()->None:
    question = "agent memory"
    
    res = question_router.invoke({"question": question})

    assert res.datasource == 'vectorstore'

def test_router_to_websearch()->None:
    question = "how to make dosa"
    
    res = question_router.invoke({"question": question})

    assert res.datasource == 'websearch'