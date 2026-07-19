from agents import researcher, supervisor, writer
from state import MessagesState
from langgraph.graph import END, START, StateGraph


g = StateGraph(MessagesState)

# Adding nodes
g.add_node("supervisor",supervisor)
g.add_node("researcher",researcher)
g.add_node("writer",writer)


# Start node
g.add_edge(START,"supervisor")


# Adding return edges to supervisor
g.add_edge("writer","supervisor")
g.add_edge("researcher","supervisor")

# Supervisor routes
def route(state):
    return state["next"]

g.add_conditional_edges("supervisor",route,{
    "researcher":"researcher",
    "writer":"writer",
    "FINISH":END
})


graph = g.compile()