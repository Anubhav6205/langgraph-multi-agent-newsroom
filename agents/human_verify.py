from langgraph.types import interrupt, Command
from langgraph.graph import END
from langchain.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from state import MessagesState


class Route(BaseModel):
    accept: bool = Field(description="True only if user's input indicates acceptance, else false")
    comments: str = Field(description="If user declines, their comments and suggestions to patch")


verify_model = init_chat_model("gpt-5-mini", temperature=0).with_structured_output(Route)

VERIFY_SYS = SystemMessage(
    "You are an assistant. Read the human's input and determine if they approve the draft. "
    "Set accept=True if they approve, False if they decline. "
    "If they decline, capture their suggestions cleanly in comments."
)


def human_verift_agent(state: MessagesState):
    human_input = interrupt({
        "draft": state["draft"],
        "question": "Approve? If no, please provide suggestions."
    })

    response = verify_model.invoke([
        VERIFY_SYS,
        HumanMessage(content=f"Human input: {human_input}")
    ])

    if response.accept:
        return Command(goto=END, update={"approved": True})

    return Command(
        goto="writer",
        update={
            "approved": False,
            "final_pass": True,
            "messages": [HumanMessage(content=f"Human reviewer suggestions: {response.comments}")]
        }
    )
