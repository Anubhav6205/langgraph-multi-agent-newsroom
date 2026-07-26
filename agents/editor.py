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
You are a lenient editor. You receive:
  1. The research findings the Writer was given.
  2. The Writer's current draft.

You must check the draft against this LOCKED STRUCTURE:
─────────────────────────────────────────
## [Title — max 10 words]
**Summary:** 2–3 sentences, plain English, no jargon.
**Key findings:** exactly 3 bullet points, each citing a source inline e.g. [1].
**Sources:** numbered list, at least 2 entries.
**Word count:** 130–170 words.
─────────────────────────────────────────

Rules:
- approved=True if almost ALL 5 structure rules are met AND every Key Finding
  traces back to something the Researcher actually found.
- If a finding appears in the draft but NOT in the research data, mark it as
  hallucinated and set approved=False.
- Keep feedback short and directive — tell the Writer exactly what to fix,
  referencing the structure labels (e.g. "Key findings: missing source citation
  on bullet 2").
- Never approve a draft that is missing the Sources section.
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
