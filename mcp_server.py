from fastmcp import FastMCP
from agent import Query, ask_agent

mcp = FastMCP("Academic Server")
long_term_memory = {} #temporary memory for agent; need to replace with database

@mcp.tool()
async def recommend_courses(question: str):
    """Recommend courses based on a question."""
    # Use the FastAPI logic directly
    query = Query(question=question)
    # Call the FastAPI endpoint logic directly (not via HTTP)
    response = await ask_agent(query)
    # Return the answer part of the response
    return response.body if hasattr(response, 'body') else response

@mcp.tool()
async def web_search(query: str):
    """Search the web for information."""

@mcp.tool()
async def save_memory(user_id: str, content: str):
    """Save a note or memory for a user."""
    long_term_memory.setdefault(user_id, []).append(content)
    return "Memory saved!"

@mcp.tool()
async def recall_memory(user_id: str):
    """Recall all notes or memories for a user."""
    return long_term_memory.get(user_id, [])

if __name__ == "__main__":
    mcp.run(transport="streamable-http",
        host="0.0.0.0",
        port=8001,
        path="/")  # run separately from FastAPI app
