from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from rich.console import Console

console = Console()

class MessagesState(TypedDict):
    """State for the agents"""
    messages: Annotated[list[AnyMessage], operator.add]
    next: Annotated[int, "This tells the next action to supervisor"]



class ResearchState(TypedDict):
    "State for research agent"
    messages: Annotated[list[AnyMessage],operator.add]
    query: str
    docs: list
    relevant: bool
