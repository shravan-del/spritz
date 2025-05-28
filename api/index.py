from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import aiohttp
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Academic Advising API",
    description="An AI-powered academic advising system using MCP for course recommendations",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    context: Optional[Dict] = {}

class Course(BaseModel):
    code: str
    name: str
    prerequisites: List[str]
    difficulty: str
    success_rate: float
    topics: List[str]

class AdvisingResponse(BaseModel):
    recommended_courses: List[Course]
    explanation: str
    next_steps: List[str]

async def get_mcp_recommendation(question: str, context: Dict) -> Dict:
    """Get course recommendations from MCP server."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:50051/request",
                json={
                    "question": question,
                    "context": context
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail="Error getting recommendations from MCP server"
                    )
                return await response.json()
    except Exception as e:
        logger.error(f"Error communicating with MCP server: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recommendations: {str(e)}"
        )

@app.post("/api/ask")
async def ask_question(request: QuestionRequest) -> JSONResponse:
    """Handle course recommendation requests."""
    try:
        # Get recommendations from MCP server
        mcp_response = await get_mcp_recommendation(
            request.question,
            request.context
        )
        
        if mcp_response.get("status") != "success":
            raise HTTPException(
                status_code=500,
                detail=mcp_response.get("error", "Unknown error occurred")
            )
            
        return JSONResponse(content=mcp_response["response"])
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to process request",
                "detail": str(e)
            }
        )

@app.get("/api/health")
async def health_check():
    """Check if the API and MCP server are healthy."""
    try:
        # Try to connect to MCP server
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:50051/health") as response:
                mcp_health = response.status == 200
    except:
        mcp_health = False

    return {
        "status": "healthy" if mcp_health else "degraded",
        "api_status": "healthy",
        "mcp_status": "healthy" if mcp_health else "unavailable"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 