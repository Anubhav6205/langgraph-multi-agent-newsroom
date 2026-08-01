# Create llm instance
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

from state import MessagesState
from pydantic import BaseModel, Field
from logger import section, log_in, log_out

class EditorOutput(BaseModel):
    approved: bool = Field(description="True if draft meets all structure rules, False if changes are needed")
    feedback: str = Field(description="Specific fixes for the writer. Empty string if approved.")

editor_model = init_chat_model("gpt-5-mini").with_structured_output(EditorOutput)
EDITOR_SYS = SystemMessage(content="""
You are a supportive editor. You receive:
  1. The research findings the Writer was given.
  2. The Writer's current draft.

Approve the draft if it is broadly well-structured and factually grounded.
Minor formatting deviations are fine — focus on substance over form.

Guidelines (not hard rules):
- A clear title, a short summary, and key findings are present
- Findings are generally traceable to the research — do not flag minor wording differences
- Sources are mentioned somewhere in the draft
- Word count is roughly in range (100–200 words is acceptable)

Rules:
- Set approved=True if the draft is good enough to show a human reader
- Only set approved=False if there is a clear factual error or a major section is entirely missing
- If approved=False, give one short specific fix — do not list multiple issues
""")


# define method to call the llm instance
def editor_agent(state: MessagesState):
    section("editor.py :: editor_agent", "bold magenta")
    log_in("revisions remaining", state["revisions"])
    log_in("draft[:120]", state["draft"][:120])

    researches = "\n".join(
        [
            m.content
            for m in state["messages"]
            if hasattr(m, "name") and m.name == "researcher"
        ]
    )
    log_in("researcher messages used", len([m for m in state["messages"] if hasattr(m, "name") and m.name == "researcher"]))
    log_in("researches[:120]", researches[:120])

    response = editor_model.invoke(
        [EDITOR_SYS]
        + [
            HumanMessage(
                content=f"Research:\n{researches}\n\nDraft:\n{state['draft']}"
            )
        ]
    )

    critique = "" if response.approved else response.feedback
    summary = f"approved={response.approved} | feedback: {response.feedback[:80]}"
    log_out("approved", response.approved)
    log_out("critique[:80]", critique[:80] if critique else "(empty — approved)")
    log_out("revisions after decrement", state["revisions"] - 1)
    return {
        "messages": [AIMessage(content=summary, name="editor")],
        "critique": critique,
        "revisions": state["revisions"] - 1,
    }
