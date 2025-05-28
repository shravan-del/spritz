import asyncio
from fastmcp.client import Client

async def main():
    async with Client("http://localhost:8001/mcp") as client:
        while True:
            question = input("Ask your advising question (or type 'exit'): ")
            if question.lower() == "exit":
                break
            try:
                # Call the recommend_courses tool
                result = await client.call_tool("recommend_courses", {"question": question})
                print("🔍 MCP Answer:\n", result)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
