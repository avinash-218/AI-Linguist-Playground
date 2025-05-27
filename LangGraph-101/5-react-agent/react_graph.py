from dotenv import load_dotenv

load_dotenv()

from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import START, END, StateGraph

from nodes import reason_node, act_node
from react_state import AgentState

REASON_NODE = "reason_node"
ACT_NODE = "act_node"
MAX_ITERATIONS = 5

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish) or state['iteration'] > MAX_ITERATIONS:
        return END
    else:
        return ACT_NODE

graph = StateGraph(AgentState)

graph.add_node(REASON_NODE, reason_node)
graph.add_node(ACT_NODE, act_node)

graph.add_edge(START, REASON_NODE)
graph.add_conditional_edges(REASON_NODE, should_continue)
graph.add_edge(ACT_NODE, REASON_NODE)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

result = app.invoke(
    {
        "input": "How many days ago was the latest SpaceX launch?",
        "agent_outcome": None,
        "intermediate_steps": [],
        "iteration": 0
    }
)

print(result)

print(result["agent_outcome"])
