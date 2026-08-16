import json

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langgraph.types import Command

load_dotenv()


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from graph import graph

app = FastAPI(title="Research API")


class ResearchRequest(BaseModel):
    task: str
    thread_id: str = "default"


class ResearchResponse(BaseModel):
    draft: str | None = None
    error: str | None = None


@app.post("/details", response_model=ResearchResponse)
async def get_details(req: ResearchRequest):
    try:
        config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 25}
        response = graph.invoke(
            {
                "messages": [HumanMessage(content=req.task)],
                "revisions": 2,
                "final_pass": False,
            },
            config,
        )

        return ResearchResponse(draft = str(response.get("draft")), error=str(response.get("error")))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/details/stream")
async def stream_details(req: ResearchRequest):
    try:
        async def token_generator():
            config = {"configurable":{"thread_id":"id_stream"},"recursion_limit":25}
            async for event in graph.astream_events(
                {
                    "messages":[HumanMessage(content=req.task)],
                    "revisions":2,
                    "final_pass":False
                },
                config,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    node = event.get("metadata",{}).get("langgraph_node","")
                    if node == "writer":
                        chunk = event["data"]["chunk"].content
                        if chunk:
                            yield f"data: {json.dumps({'token':chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))