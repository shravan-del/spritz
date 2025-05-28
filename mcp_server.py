from fastmcp import MCPServer, MCPRequest
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseAdvisorMCP(MCPServer):
    def __init__(self):
        super().__init__()
        self.course_history: Dict[str, List[Dict]] = {}
        self.feedback_data: Dict[str, List[Dict]] = {}
        
    async def handle_request(self, request: MCPRequest) -> Dict:
        """Handle incoming course recommendation requests."""
        try:
            # Extract question and context
            question = request.content.get("question", "")
            context = request.content.get("context", {})
            
            # Log the request for learning
            self.log_request(question, context)
            
            # Generate response based on historical data and current context
            response = await self.generate_recommendation(question, context)
            
            # Update learning data
            self.update_learning_data(question, response)
            
            return {
                "status": "success",
                "response": response,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request.request_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request.request_id
                }
            }

    def log_request(self, question: str, context: Dict) -> None:
        """Log requests for learning purposes."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "question": question,
            "context": context
        }
        self.course_history.setdefault(question, []).append(log_entry)

    async def generate_recommendation(self, question: str, context: Dict) -> Dict:
        """Generate course recommendations using historical data and context."""
        # Sample course data (replace with your actual course database)
        courses = [
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

        # Filter courses based on context and question
        recommended_courses = []
        for course in courses:
            if self.is_course_relevant(course, question, context):
                recommended_courses.append(course)

        return {
            "recommended_courses": recommended_courses[:3],  # Top 3 recommendations
            "explanation": self.generate_explanation(recommended_courses),
            "next_steps": self.suggest_next_steps(recommended_courses)
        }

    def is_course_relevant(self, course: Dict, question: str, context: Dict) -> bool:
        """Determine if a course is relevant to the student's question and context."""
        # Add your course relevance logic here
        return True  # Placeholder - implement actual relevance checking

    def generate_explanation(self, courses: List[Dict]) -> str:
        """Generate a human-readable explanation for the recommendations."""
        if not courses:
            return "No courses match your current criteria."
        
        return f"Based on your interests and academic background, these {len(courses)} courses " \
               f"would be most beneficial for your computer science journey."

    def suggest_next_steps(self, courses: List[Dict]) -> List[str]:
        """Suggest next steps based on the recommended courses."""
        steps = [
            "Review the prerequisites for each recommended course",
            "Check the course schedules and availability",
            "Consider your current workload and time commitment",
            "Reach out to academic advisors for additional guidance"
        ]
        return steps

    def update_learning_data(self, question: str, response: Dict) -> None:
        """Update the learning data based on responses and feedback."""
        timestamp = datetime.now().isoformat()
        learning_entry = {
            "timestamp": timestamp,
            "question": question,
            "response": response
        }
        self.feedback_data.setdefault(question, []).append(learning_entry)

if __name__ == "__main__":
    # Create and run the MCP server
    server = CourseAdvisorMCP()
    asyncio.run(server.run(host="0.0.0.0", port=50051))
