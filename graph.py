from agents.researcher import researcher_agent

from agents.supervisor import supervistor_agent
from agents.writer import writer_agent
from state import MessagesState
from langgraph.graph import END, START, StateGraph
from logger import section, log_in

g = StateGraph(MessagesState)

# Adding nodes
g.add_node("supervisor",supervistor_agent)
g.add_node("researcher",researcher_agent)
g.add_node("writer",writer_agent)


# Start node
g.add_edge(START,"supervisor")


# Adding return edges to supervisor
g.add_edge("writer","supervisor")
g.add_edge("researcher","supervisor")

# Supervisor routes
def route(state):
    section("graph.py :: route", "bold white on dark_blue")
    next_node = state["next"]
    log_in("state['next']", next_node)
    log_in("state['messages'] count", len(state["messages"]))
    return next_node

g.add_conditional_edges("supervisor",route,{
    "researcher":"researcher",
    "writer":"writer",
    "FINISH":END
})


graph = g.compile()