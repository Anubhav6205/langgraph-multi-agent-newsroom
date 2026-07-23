from os import name

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph
from state import MessagesState
from logger import section, log_messages, log_out
from pydantic import BaseModel, Field

class Route(BaseModel):
    brief : str = Field(description = "Draft concise brief of the transcript")
    cause: str = Field(description="Explain in strictly less than 20 words why you took the descision.")


writer_model = init_chat_model("gpt-5-mini",temperature=0).with_structured_output(Route)

WRITER_SYS = SystemMessage(
    "You are a professional writer. Draft concise brief of the transcript.Use no tools"
)

def writer_agent(state: MessagesState):
    section("writer.py :: writer_agent", "bold red")
    log_messages(state["messages"])

    response = writer_model.invoke([WRITER_SYS]+state["messages"])

    log_out("response type", type(response).__name__)
    log_out("content[:120]", str(response.content)[:120])
    return {
        "messages": [AIMessage(content=response.cause, name="writer")]
    }

