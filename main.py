import logging
import os
from typing import Annotated
from typing import TypedDict

import dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

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
    thread_id: str | None = None


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
graph_builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})

# Connect tool back to chatbot
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Add checkpoint
memory = MemorySaver()

# Compile graph
agent = graph_builder.compile(checkpointer=memory)


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logging.info(f"Received message: {request.message}")

        config = {"configurable": {"thread_id": request.thread_id}}

        # Get existing state or create new one
        try:
            state = agent.get_state(config)
            messages = state.values["messages"]
        except Exception as e:
            logging.info(f"No existing state, starting fresh: {e}")
            messages = []

        current_messages = messages + [("user", request.message)]

        # Invoke agent with current messages
        response = await agent.ainvoke(
            {"messages": current_messages},
            config=config,
        )

        last_message = response["messages"][-1]
        response_content = last_message.content

        # Check if response contains search request
        if "NEED_SEARCH:" in response_content:
            # Extract search query and return it to client
            query = response_content.split("NEED_SEARCH:")[1].strip()
            logging.info(f"Search query: {query}")
            return {"needs_search": True, "search_query": query}

        # If we have search results, include them in the prompt
        if request.search_results:
            logging.info("Search results provided, adding to prompt")
            search_message = f"Search results: {request.search_results}\n\nUser question: {request.message}"
            current_messages = messages + [("user", search_message)]

            final_response = await agent.ainvoke(
                {"messages": current_messages},
                config=config,
            )

            response_content = final_response["messages"][-1].content
            logging.info(f"Final response: {response_content}")
            return {"response": response_content, "needs_search": False}

        logging.info(f"Final response: {response_content}")
        return {"response": response_content, "needs_search": False}

    except Exception as e:
        logging.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
