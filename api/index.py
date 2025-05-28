from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import os
from typing import List, Optional

app = FastAPI(
    title="Academic Advising API",
    description="An API for providing academic advising using course data",
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

class CourseRecommendation(BaseModel):
    code: str
    gpa: str
    professors: List[str]
    difficulty: str

class AdvisingResponse(BaseModel):
    recommended_courses: List[CourseRecommendation]
    key_points: List[str]
    summary: str

# Sample course data (in production, this would come from a database)
SAMPLE_COURSES = [
    {
        "code": "CS101",
        "gpa": "3.8",
        "professors": ["Dr. Smith (4.5)", "Dr. Johnson (4.2)"],
        "difficulty": "Easy"
    },
    {
        "code": "CS201",
        "gpa": "3.5",
        "professors": ["Dr. Williams (4.3)"],
        "difficulty": "Moderate"
    },
    {
        "code": "CS301",
        "gpa": "3.2",
        "professors": ["Dr. Brown (4.0)", "Dr. Davis (4.1)"],
        "difficulty": "Challenging"
    }
]

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    try:
        # For demo purposes, always return some sample recommendations
        response = {
            "recommended_courses": SAMPLE_COURSES,
            "key_points": [
                "Selected courses with strong student performance history",
                "Included a mix of difficulty levels",
                "All recommended professors have good ratings"
            ],
            "summary": "These courses provide a balanced academic path with experienced professors and proven student success rates."
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# This is for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 