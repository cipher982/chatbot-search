import logging

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import os
import dotenv

# Add these imports
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing import TypedDict


dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")



def add_messages(current: list, new: list) -> list:
    """Helper function for type hints that combines two message lists"""
    return current + new


class ChatRequest(BaseModel):
    message: str
    search_results: dict | None = None

class State(TypedDict):
    messages: Annotated[list, add_messages]


def tools_condition(state: State) -> str:
    """Determines if the response requires a tool call"""
    last_message = state["messages"][-1]
    if "NEED_SEARCH:" in last_message:
        return "tools"
    return END




def web_search(query: str) -> str:
    """This tool doesn't actually perform the search, it signals to the frontend
    that a search is needed"""
    return f"NEED_SEARCH:{query}"


llm = ChatOpenAI(
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
)

tools = [
    Tool(
        name="web_search",
        func=web_search,
        description="Search the web for current information. Input should be a search query.",
    )
]



graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.bind_tools(tools).invoke(state["messages"])]}

# Create tool node to handle searches
tool_node = ToolNode(tools=tools)

# Add nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Add conditional routing
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", END: END}
)

# Connect tool back to chatbot
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile graph
agent = graph_builder.compile()



@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logging.info(f"Received message: {request.message}")
        # First pass: LLM decides if it needs to search
        response = await agent.ainvoke({"messages": [("user", request.message)]})
        response = response["output"]

        # Check if response contains search request
        if "NEED_SEARCH:" in response:
            # Extract search query and return it to client
            query = response.split("NEED_SEARCH:")[1].strip()
            logging.info(f"Search query: {query}")
            return {"needs_search": True, "search_query": query}

        # If we have search results, include them in the prompt
        if request.search_results:
            logging.info("Search results provided, adding to prompt")
            final_response = await agent.arun(
                f"Search results: {request.search_results}\n\nUser question: {request.message}"
            )
            logging.info(f"Final response: {final_response}")
            return {"response": final_response, "needs_search": False}

        logging.info(f"Final response: {response}")
        return {"response": response, "needs_search": False}

    except Exception as e:
        logging.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
