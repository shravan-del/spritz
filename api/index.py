from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Academic Advising API",
    description="An AI-powered academic advising system for course recommendations",
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

# Sample course database
COURSES = [
    {
        "code": "CS101",
        "name": "Introduction to Computer Science",
        "prerequisites": [],
        "difficulty": "Beginner",
        "success_rate": 0.85,
        "topics": ["programming basics", "algorithms", "data structures"]
    },
    {
        "code": "CS201",
        "name": "Data Structures and Algorithms",
        "prerequisites": ["CS101"],
        "difficulty": "Intermediate",
        "success_rate": 0.75,
        "topics": ["advanced algorithms", "complex data structures"]
    },
    {
        "code": "CS301",
        "name": "Software Engineering",
        "prerequisites": ["CS201"],
        "difficulty": "Advanced",
        "success_rate": 0.70,
        "topics": ["software design", "project management", "testing"]
    }
]

def get_course_recommendations(question: str, context: Dict) -> Dict:
    """Generate course recommendations based on question and context."""
    # Simple keyword-based matching (can be enhanced with NLP)
    keywords = question.lower().split()
    
    recommended_courses = []
    for course in COURSES:
        # Check if course topics match any keywords
        relevant = any(
            keyword in " ".join(course["topics"]).lower() 
            for keyword in keywords
        )
        if relevant:
            recommended_courses.append(course)
    
    # If no matches, return all courses
    if not recommended_courses:
        recommended_courses = COURSES[:2]  # Return first 2 courses as default
    
    return {
        "recommended_courses": recommended_courses,
        "explanation": f"Based on your interest in {', '.join(keywords)}, " 
                      f"these {len(recommended_courses)} courses would be beneficial.",
        "next_steps": [
            "Review the prerequisites for each recommended course",
            "Check the course schedules and availability",
            "Consider your current workload and time commitment",
            "Reach out to academic advisors for additional guidance"
        ]
    }

@app.post("/api/ask")
async def ask_question(request: QuestionRequest) -> JSONResponse:
    """Handle course recommendation requests."""
    try:
        # Generate recommendations
        response = get_course_recommendations(
            request.question,
            request.context
        )
        
        return JSONResponse(content=response)
        
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
    """Check if the API is healthy."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 