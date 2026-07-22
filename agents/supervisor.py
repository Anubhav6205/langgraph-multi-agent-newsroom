from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from pydantic import BaseModel, Field
from state import MessagesState
from logger import section, log_messages, log_out

class Route(BaseModel):
    """Who will act next"""
    next: Literal["researcher","writer","FINISH"] = Field(description="The next action will be taken up by which agent, or FINISH if the task is completed.")



supervisor_llm = init_chat_model("gpt-5-mini",temperature=0).with_structured_output(Route)

SUPERVISOR_SYS = SystemMessage(
    "You are a supervisor of a newsroom and have two specialist under you: A researcher (gathers sourced facts via web search) and writer (drafts a brief from those facts). Read the conversation so far and decide who acts NEXT: \n"
    "- 'researcher' if more facts /sources are needed before writing\n"
    "- 'writer' once enough facts are gathered to draft and revise the brief"
    "- 'FINISH' when a complete, well sources brief exist in the transcript"
    "Invoke Researcher for details on a topic, then invoke writer to write about the topic. Route one step at a time. Do not work yourself, only decide who is next actor"
)

def supervistor_agent(state: MessagesState):
    section("supervisor.py :: supervistor_agent", "bold blue")
    log_messages(state["messages"])

    respose = supervisor_llm.invoke([SUPERVISOR_SYS]+ state["messages"])

    log_out("supervisor decision → next", respose.next)
    return {"next": respose.next}