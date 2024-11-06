import asyncio
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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(funcName)s:%(lineno)d] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)

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
        self._waiting_for_response = False

    async def fetch_url(self, query: str) -> str:
        try:
            logging.info(f"Sending URL request: {query}")
            await self._websocket.send_json({"type": "url_request", "query": query})
            logging.info("Waiting for response...")
            response = await asyncio.wait_for(self._websocket.receive_json(), timeout=10.0)
            logging.info(f"Received response: {response}")
            return response["results"]
        except asyncio.TimeoutError:
            logging.error("URL request timed out waiting for response")
            raise HTTPException(status_code=504, detail="URL request timed out")

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.fetch_url,
            name="url_request",
            description="Ftch content from URLs. Input should be a URL.",
            coroutine=self.fetch_url,
        )


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = websocket
    try:
        while True:
            await asyncio.sleep(1)
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
        url_tool = URLRequestTool(websocket)
        tool = url_tool.get_tool()
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        llm_with_tools = llm.bind_tools([tool])

        # Process message
        response = await llm_with_tools.ainvoke([{"role": "user", "content": request.message}])
        logging.info(f"Full Response: {response}")

        # If there are tool calls, execute them and get final response
        if response.tool_calls:
            logging.info(f"Tool Calls: {response.tool_calls}")
            tool_call = response.tool_calls[0]
            url = tool_call["args"]["query"]
            url_result = await url_tool.fetch_url(url)

            # Get final response with tool results
            input_messages = [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {"role": "tool", "content": url_result, "tool_call_id": tool_call["id"]},
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
