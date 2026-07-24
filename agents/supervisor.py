from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from state import MessagesState
from logger import section, log_in, log_messages, log_out

class Route(BaseModel):
    """Who will act next"""
    next: Literal["researcher","writer","FINISH","editor"] = Field(description="The next action will be taken up by which agent, or FINISH if the task is completed.")
    cause: str = Field(description="Explain in strictly less than 20 words why you took the descision.")



supervisor_llm = init_chat_model("gpt-5-mini",temperature=0).with_structured_output(Route)

SUPERVISOR_SYS = SystemMessage(
    "You are a supervisor of a newsroom. Route one step at a time using this STRICT sequence based on who spoke last:\n"
    "- No agent has spoken yet → route to 'researcher'\n"
    "- Last speaker was 'researcher' → route to 'writer'\n"
    "- Last speaker was 'writer' → route to 'editor'\n"
    "- Last speaker was 'editor' AND revision counter > 0 → route to 'writer'\n"
    "- Last speaker was 'editor' AND revision counter == 0 → route to 'FINISH'\n"
    "Do not deviate from this sequence. Do not route researcher again once writing has started."
    )

def supervistor_agent(state: MessagesState):
    section("supervisor.py :: supervistor_agent", "bold blue")
    log_in("revisions remaining", state["revisions"])
    log_messages(state["messages"])

    respose = supervisor_llm.invoke([SUPERVISOR_SYS]+ state["messages"] + [HumanMessage(content=f"Revision counter: {state['revisions']}")])

    log_out("supervisor decision → next", respose.next)
    log_out("cause", respose.cause)
    return {"next": respose.next}