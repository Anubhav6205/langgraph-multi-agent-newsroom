# W6 Agents 2 — Multi-Agent Newsroom

A multi-agent system built with LangGraph where a Supervisor routes tasks between a Researcher and a Writer agent.

---

## Architecture

### Outer Graph (`graph.py`)

```
START → supervisor
            ↓ route()
            ├── researcher
            ├── writer
            └── FINISH → END

writer → supervisor (loop back)
researcher → supervisor (loop back)
```

### Inner Researcher Graph (`agents/researcher.py`)

The researcher uses a **Corrective RAG** pattern — it grades its own retrieval and rewrites the query if the results are irrelevant.

```
START
  ↓
_agent  (LLM reads messages + RESEARCHER_SYS prompt, decides to call a tool)
  ↓ tools_condition
  ├── END  (no tool call → LLM has synthesized a final answer)
  └── ToolNode  (actually executes search_kb or web_search, appends ToolMessage to messages)
         ↓
     extract_docs  (reads last ToolMessage from messages, writes to state["docs"])
         ↓
       grade  (reads state["query"] + state["docs"], writes state["relevant"])
         ↓ route_grade
         ├── _agent  (relevant=True → LLM synthesizes final answer → tools_condition → END)
         └── rewrite  (relevant=False → rewrites query, appends new HumanMessage to messages)
                ↓
             _agent  (retry — LLM sees new message, calls search_kb again)
                ↓ (cycle repeats)
```

---

## Node Responsibilities

| Node | Reads from State | Writes to State |
|---|---|---|
| `_agent` | `messages` | `messages` (AIMessage with or without tool_calls) |
| `ToolNode` | `messages` (last AIMessage tool_calls) | `messages` (ToolMessage with tool output) |
| `extract_docs` | `messages` (last ToolMessage) | `docs` |
| `grade_agent` | `query`, `docs` | `relevant` |
| `rewrite_query_agent` | `query` | `query` (new value), `messages` (new HumanMessage) |

---

## State

### `MessagesState` — outer graph

```python
messages: Annotated[list[AnyMessage], operator.add]
next: str  # supervisor sets this to route to "researcher", "writer", or "FINISH"
```

### `ResearchState` — inner researcher graph

```python
messages: Annotated[list[AnyMessage], operator.add]
query: str    # the current search query; rewrite_query_agent updates this
docs: list    # raw retrieval results; extract_docs populates this
relevant: bool  # grade_agent sets this; used by route_grade to branch
```

---

## Key Design Decisions

**Why `extract_docs` is a separate node**
`ToolNode` is a LangGraph built-in — it only writes to `messages`. It has no awareness of custom state fields like `docs`. The `extract_docs` node bridges the gap: it reads from `messages` and writes to `docs` so `grade_agent` can operate on clean structured data.

**Why `rewrite_query_agent` must write to `messages`**
`_agent` only reads `state["messages"]` — it never reads `state["query"]`. If `rewrite_query_agent` only updated `state["query"]`, `_agent` would re-run with the same conversation history and repeat the same search. Writing a new `HumanMessage` into `messages` gives the LLM an explicit signal to search differently.

**Why `relevant=True` routes back to `_agent`**
When `grade_agent` approves the docs, `_agent` re-runs with the full conversation history (original query + tool call + tool result). The LLM synthesizes a final answer with no tool call, so `tools_condition` routes to `END`. The outer `researcher_agent` wrapper then picks up the last message and returns it to the supervisor.

---

## Tools

| Tool | Source | Purpose |
|---|---|---|
| `search_kb` | `tools/search_kb.py` | Searches local Chroma vector store (tries first) |
| `web_search` | `tools/web_search.py` | Falls back to web if KB yields nothing useful |
