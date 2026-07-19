from os import name

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph
from state import MessagesState
writer_model = init_chat_model("gpt-5-mini",temperature=0)

WRITER_SYS = SystemMessage(
    "You are a professional writer. Draft concise brief of the transcript.Use no tools"
)

def writer_agent(state: MessagesState):
    response = writer_model.invoke([WRITER_SYS]+state["message"])
    return {
        "messages": [AIMessage(content=response.content, name="writer")]
    }

