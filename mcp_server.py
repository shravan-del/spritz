from fastmcp import FastMCP
from agent import Query, ask_agent

mcp = FastMCP("Academic Server")

@mcp.tool()
async def recommend_courses(question: str):
    """Recommend courses based on a question."""
    # Use the FastAPI logic directly
    query = Query(question=question)
    # Call the FastAPI endpoint logic directly (not via HTTP)
    response = await ask_agent(query)
    # Return the answer part of the response
    return response.body if hasattr(response, 'body') else response

if __name__ == "__main__":
    mcp.run(transport="streamable-http",
        host="0.0.0.0",
        port=8001,
        path="/")  # run separately from FastAPI app
