import json
import logging

import dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store active WebSocket connections
connections = {}


class ChatRequest(BaseModel):
    message: str
    websocket_id: str


class URLRequestTool:
    def __init__(self, websocket: WebSocket):
        self._websocket = websocket

    async def _search(self, query: str) -> str:
        """Search or fetch content from URLs. Input should be a URL."""
        await self._websocket.send_json({"type": "search_request", "query": query})
        response = await self._websocket.receive_text()
        return json.loads(response)["results"]

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self._search,
            name="url_request",
            description="Search or fetch content from URLs. Input should be a URL.",
            coroutine=self._search,
        )


# url_request_tool = URLRequestTool()
# llm = ChatOpenAI(
#     temperature=0,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     model="gpt-4o-mini",
# )
# llm_with_tools = llm.bind_tools([url_request_tool])


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    finally:
        del connections[client_id]


@app.post("/chat")
async def chat(request: ChatRequest):
    websocket = connections.get(request.websocket_id)
    if not websocket:
        raise HTTPException(status_code=400, detail="No active connection")

    try:
        # Create tool instance with current websocket
        url_tool = URLRequestTool(websocket).get_tool()

        # Initialize LLM with tool
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        llm_with_tools = llm.bind_tools([url_tool])

        # Process message
        response = await llm_with_tools.ainvoke([{"role": "user", "content": request.message}])
        return {"response": response.content}

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(RequestValidationError)  # Add this handler
async def validation_exception_handler(request, exc):
    logging.error(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
