from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SpritzAI: Academic Advisor",
    description="AI-powered academic advising system for Virginia Tech students",
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
    student_info: Optional[Dict] = {
        "major": "",
        "year": "",
        "interests": [],
        "completed_courses": []
    }

class Professor(BaseModel):
    name: str
    rating: float
    review_count: int
    difficulty: float

class Course(BaseModel):
    code: str
    name: str
    credits: int
    prerequisites: List[str]
    difficulty: str
    gpa_avg: float
    a_percentage: float
    professors: List[Professor]
    topics: List[str]
    semester_availability: List[str]
    major_requirements: List[str]
    student_reviews: List[str]

class AdvisingResponse(BaseModel):
    recommended_courses: List[Course]
    explanation: str
    next_steps: List[str]
    pathway_plan: Optional[Dict[str, List[str]]]
    additional_resources: List[Dict[str, str]]

# Enhanced course database with VT-specific information
COURSES = [
    {
        "code": "CS1114",
        "name": "Introduction to Software Design",
        "credits": 3,
        "prerequisites": [],
        "difficulty": "Moderate",
        "gpa_avg": 3.2,
        "a_percentage": 45.0,
        "professors": [
            {"name": "Dr. Smith", "rating": 4.5, "review_count": 120, "difficulty": 3.2},
            {"name": "Dr. Johnson", "rating": 4.2, "review_count": 85, "difficulty": 3.5}
        ],
        "topics": ["Python", "basic programming concepts", "problem-solving", "algorithms"],
        "semester_availability": ["Fall", "Spring"],
        "major_requirements": ["CS", "CpE"],
        "student_reviews": [
            "Great intro course for learning Python",
            "Challenging but rewarding",
            "Take it with Dr. Smith if possible"
        ]
    },
    {
        "code": "CS2114",
        "name": "Software Design & Data Structures",
        "credits": 3,
        "prerequisites": ["CS1114"],
        "difficulty": "Challenging",
        "gpa_avg": 3.0,
        "a_percentage": 35.0,
        "professors": [
            {"name": "Dr. Williams", "rating": 4.3, "review_count": 95, "difficulty": 3.8}
        ],
        "topics": ["Java", "data structures", "object-oriented programming"],
        "semester_availability": ["Fall", "Spring", "Summer"],
        "major_requirements": ["CS", "CpE"],
        "student_reviews": [
            "Essential course for CS fundamentals",
            "Heavy workload but very informative"
        ]
    }
]

def get_course_recommendations(question: str, context: Dict, student_info: Dict) -> Dict:
    """Generate personalized course recommendations based on question and student context."""
    # Convert question to lowercase for better matching
    question_lower = question.lower()
    
    # Extract key information from student info
    year = student_info.get("year", "").lower()
    major = student_info.get("major", "").lower()
    completed_courses = set(student_info.get("completed_courses", []))
    
    # Keywords for first-year focus
    first_year_keywords = ["first year", "freshman", "begin", "start", "new"]
    is_first_year_query = any(keyword in question_lower for keyword in first_year_keywords)
    
    # Filter courses based on various criteria
    recommended_courses = []
    for course in COURSES:
        # Skip courses already completed
        if course["code"] in completed_courses:
            continue
            
        # Check prerequisites
        prereqs_met = all(prereq in completed_courses for prereq in course["prerequisites"])
        
        # Check if course matches query context
        is_relevant = (
            (is_first_year_query and not course["prerequisites"]) or  # First-year appropriate
            any(topic.lower() in question_lower for topic in course["topics"]) or  # Topic match
            course["code"].lower() in question_lower or  # Direct code reference
            course["name"].lower() in question_lower  # Course name match
        )
        
        if is_relevant and (is_first_year_query or prereqs_met):
            recommended_courses.append(course)
    
    # Sort courses by relevance (simplified version)
    recommended_courses.sort(key=lambda x: (-x["gpa_avg"], -x["a_percentage"]))
    
    # Generate pathway plan for first-year students
    pathway_plan = None
    if is_first_year_query:
        pathway_plan = {
            "First Semester": ["CS1114", "MATH1225", "ENGL1105"],
            "Second Semester": ["CS2114", "MATH1226", "ENGL1106"],
            "Summer Options": ["CS2104", "MATH2114"]
        }
    
    # Compile response
    return {
        "recommended_courses": recommended_courses[:3],  # Top 3 recommendations
        "explanation": generate_explanation(recommended_courses, question, student_info),
        "next_steps": generate_next_steps(recommended_courses),
        "pathway_plan": pathway_plan,
        "additional_resources": [
            {"type": "Academic Advising", "link": "https://cs.vt.edu/undergraduate/advising.html"},
            {"type": "Course Catalog", "link": "https://cs.vt.edu/undergraduate/courses.html"},
            {"type": "Prerequisites Chart", "link": "https://cs.vt.edu/undergraduate/prerequisites.html"}
        ]
    }

def generate_explanation(courses: List[Dict], question: str, student_info: Dict) -> str:
    """Generate a detailed explanation for course recommendations."""
    if not courses:
        return "Based on your query, I couldn't find any suitable courses. Please try being more specific or consult with an academic advisor."
    
    course_names = [f"{c['code']} ({c['name']})" for c in courses]
    
    explanation = f"Based on your interest in {question}, I recommend these courses:\n\n"
    for i, course in enumerate(courses, 1):
        explanation += f"{i}. {course['code']}: {course['name']}\n"
        explanation += f"   • Average GPA: {course['gpa_avg']:.1f}\n"
        explanation += f"   • Success Rate (A grades): {course['a_percentage']}%\n"
        explanation += f"   • Best Professor: {max(course['professors'], key=lambda x: x['rating'])['name']}\n"
        explanation += f"   • Available in: {', '.join(course['semester_availability'])}\n\n"
    
    return explanation

def generate_next_steps(courses: List[Dict]) -> List[str]:
    """Generate personalized next steps based on recommended courses."""
    steps = [
        "Review the prerequisites for each recommended course",
        "Check the course schedules in HokieSPA",
        "Read detailed course reviews on Anaanu",
        "Consider your workload and time commitment",
        "Schedule a meeting with your academic advisor",
        "Join CS@VT Discord for peer advice"
    ]
    return steps

@app.post("/api/ask")
async def ask_question(request: QuestionRequest) -> JSONResponse:
    """Handle course recommendation requests."""
    try:
        # Generate recommendations
        response = get_course_recommendations(
            request.question,
            request.context,
            request.student_info
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
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 