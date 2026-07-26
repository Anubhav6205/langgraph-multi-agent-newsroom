from agents.editor import editor_agent
from agents.human_verify import human_verift_agent
from agents.researcher import researcher_agent
from agents.supervisor import supervistor_agent
from agents.writer import writer_agent
from state import MessagesState
from langgraph.graph import END, START, StateGraph
from logger import section, log_in
from langgraph.checkpoint.memory import InMemorySaver

g = StateGraph(MessagesState)

# Adding nodes
g.add_node("supervisor", supervistor_agent)
g.add_node("researcher", researcher_agent)
g.add_node("writer", writer_agent)
g.add_node("editor", editor_agent)
g.add_node("human_verify", human_verift_agent)

# Start node
g.add_edge(START, "supervisor")

# Researcher and editor always return to supervisor
g.add_edge("researcher", "supervisor")
g.add_edge("editor", "supervisor")


# Writer routes to supervisor normally, or END if human already reviewed (final_pass)
def route_writer(state):
    if state.get("final_pass"):
        return END
    return "supervisor"


g.add_conditional_edges("writer", route_writer, {"supervisor": "supervisor", END: END})


# Supervisor routes
def route(state):
    section("graph.py :: route", "bold white on dark_blue")
    next_node = state["next"]
    log_in("state['next']", next_node)
    log_in("state['messages'] count", len(state["messages"]))
    return next_node


g.add_conditional_edges(
    "supervisor",
    route,
    {
        "researcher": "researcher",
        "writer": "writer",
        "editor": "editor",
        "FINISH": "human_verify",
    },
)

checkpointer = InMemorySaver()
graph = g.compile(checkpointer=checkpointer)
