from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    """State for the agents"""
    message: Annotated[list[AnyMessage], operator.add]
    next: Annotated[int, "This tells the next action to supervisor"]