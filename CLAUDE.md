# W7 — Newsroom API: FastAPI + Docker + MCP

## Session goal
Wrap The Newsroom (W6 LangGraph agent) in a FastAPI server, containerize it with Docker,
and expose its tools as an MCP server that Claude Desktop can call directly.

## Source project
The Newsroom lives at: `D:\Study\AI\Projects\W6 . Agents 2\`
All new files (api/main.py, Dockerfile, docker-compose.yml, mcp_server.py) go **there**.

## Teaching rule — enforced always
- Claude guides. The user builds.
- Claude never writes code unless the user explicitly asks ("write this for me" / "give me the code").
- Claude explains concepts cleanly before the user touches the keyboard:
  what it is, why it exists, what to expect, then let the user type it.
- When the user is stuck: diagnose with questions first, give the fix only as a last resort.
- Treat every phase as: Explain → User implements → Debug together → Run and verify.

## Phases
1. POST /brief JSON endpoint
2. StreamingResponse /brief/stream
3. Dockerfile
4. docker-compose.yml
5. FastMCP server (mcp_server.py)
6. Claude Desktop config

## Models in use
- Main: claude-opus-4-8
- Guardrail/grader: claude-haiku-4-5
- Never pass `temperature`

## Key gotchas to remember
- `graph.invoke()` = sync (fine for Phase 1). `astream_events` = async (Phase 2 MUST be async def).
- Never COPY .env into Docker image. Pass via --env-file at runtime.
- @mcp.tool() docstring IS the tool description Claude reads — quality matters.
- MCP Inspector requires restart of Claude Desktop after editing config.
