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

    async def fetch_url(self, query: str) -> str:
        logging.info(f"tool, Fetching URL: {query}")
        """Search or fetch content from URLs. Input should be a URL."""
        await self._websocket.send_json({"type": "search_request", "query": query})
        response = await self._websocket.receive_text()
        logging.info(f"tool, Response: {response}")
        return json.loads(response)["results"]

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.fetch_url,
            name="url_request",
            description="Search or fetch content from URLs. Input should be a URL.",
            coroutine=self.fetch_url,
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
    logging.info(f"Chat Request: {request}")
    websocket = connections.get(request.websocket_id)
    if not websocket:
        raise HTTPException(status_code=400, detail="No active connection")

    try:
        # Create tool instance with current websocket
        tool = URLRequestTool(websocket).get_tool()
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        llm_with_tools = llm.bind_tools([tool])

        # Process message
        response = await llm_with_tools.ainvoke([{"role": "user", "content": request.message}])
        logging.info(f"Full Response: {response}")

        # If there are tool calls, execute them and get final response
        if response.tool_calls:
            logging.info(f"Tool Calls: {response.tool_calls}")
            tool_call = response.tool_calls[0]
            tool_input = tool_call["args"]["query"]
            tool_result = await tool.invoke(tool_input)

            # Get final response with tool results
            input_messages = [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {"role": "tool", "content": str(tool_result), "tool_call_id": tool_call.id},
            ]
            final_response = await llm_with_tools.ainvoke(input_messages)
            logging.info(f"Final Response (with tool result): {final_response}")
            return {"response": final_response.content}

        logging.info(f"Final Response (no tool calls): {response.content}")
        return {"response": response.content}

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logging.error(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")
