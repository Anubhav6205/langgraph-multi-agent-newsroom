from os import name

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph
from state import MessagesState
from logger import section, log_messages, log_out

writer_model = init_chat_model("gpt-5-mini",temperature=0)

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
        "messages": [AIMessage(content=response.content, name="writer")]
    }

