from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langgraph.graph import START, END

llm = ChatGroq(model='llama3-8b-8192')

class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str

workflow = StateGraph(State)

def categorize_experience(state: State)-> State:
    print("\nCategorizing the experience level of candidate : ")
    prompt = ChatPromptTemplate.from_template(
        "Based on the following job application, categorize the candidate as Entry-level, Mid-level, Senior-level"
        "Respond with one word either Entry-level, Mid-level, Senior-level"
        "Application : {application}")
    
    chain = prompt | llm
    experience_level = chain.invoke({"application": state["application"]}).content
    print(f"experience_level: {experience_level}")
    state["experience_level"] = experience_level
    return state

def assess_skills(state:State)->State:
    print("\nAssessing the skillset of candidate : ")
    prompt = ChatPromptTemplate.from_template(
        "Based on the job application for Python Developer, assess the candidate's skillset"
        "Respond with one word either Match or No Match"
        "Application : {application}")
    
    chain = prompt | llm
    skill_match = chain.invoke({"application": state["application"]}).content
    print(f"skill_match: {skill_match}")
    state["skill_match"] = skill_match
    return state

def schedule_interview(state: State)-> State:
    print("\nScheduling the interview : ")
    state["response"] = "Candidate has been shortlisted for an HR interview"
    return state

def escalate_to_recruiter(state: State)-> State:
    print("\nEscalting to recruiter : ")
    state["response"] = "Candidate has senior-level experence but doesn't match job skills."
    return state

def reject_application(state: State)-> State:
    print("\nSending rejection email : ")
    state["response"] = "Candidate doesn't meet JD and has been rejected"
    return state

def route_app(state:State)-> str:
    if state['skill_match'] == "Match":
        return "schedule_interview"
    else:
        if state['experience_level'] == "Senior-level":
            return "escalate_to_recruiter"
        else:
            return "reject_application"

# create graph
# add nodes
workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("assess_skills", assess_skills)
workflow.add_node("schedule_interview", schedule_interview)
workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
workflow.add_node("reject_application", reject_application)

# add edges
workflow.add_edge(START, "categorize_experience")
workflow.add_edge("categorize_experience", "assess_skills")
workflow.add_conditional_edges("assess_skills", route_app)
workflow.add_edge("schedule_interview", END)
workflow.add_edge("escalate_to_recruiter", END)
workflow.add_edge("reject_application", END)

graph = workflow.compile()

# graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

def run_candidate_screening(application: str):
    results = graph.invoke({"application": application})
    return {"experience_level": results["experience_level"],
            "skill_match": results["skill_match"],
            "response": results["response"]}

application_text = "I have 3 years of experience in software engineering with expertise in JAVA"
results = run_candidate_screening(application_text)
print("\n\nCandidate Screening Results :")
print(f'1) Application : {application_text}')
print(f'2) Experience Level : {results["experience_level"]}')
print(f'3) Skill Match : {results["skill_match"]}')
print(f'4) Response : {results["response"]}')