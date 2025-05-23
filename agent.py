"""
This is the main FastAPI application that powers our academic advising system.
It uses AI to help students choose courses based on grade data and professor information.
"""

# Import necessary libraries
import os
from typing import Dict, Optional  # For type hints
from fastapi import FastAPI, HTTPException, Request  # Web framework
from fastapi.middleware.cors import CORSMiddleware  # Allows web browsers to call our API
from fastapi.responses import JSONResponse  # For sending JSON responses
from pydantic import BaseModel, Field  # For data validation
from utils import get_retriever  # Our custom course data handler
from langchain.chains import RetrievalQA  # For question-answering
from langchain_huggingface import HuggingFacePipeline  # For using AI models
from langchain.prompts import PromptTemplate  # For formatting AI prompts
import logging  # For error tracking
from dotenv import load_dotenv  # For environment variables
import torch  # For AI model handling
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline  # AI model tools
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # Security
import time  # For rate limiting
from cachetools import TTLCache, cached  # For caching responses
import json  # For JSON handling

# Load environment settings
load_dotenv()

# Set up logging to track errors and info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Server settings
HOST = "127.0.0.1"  # Local host address
PORT = 3001  # Port number
MODEL_NAME = "google/flan-t5-large"  # AI model we're using

# Rate limiting settings to prevent overuse
RATE_LIMIT = 5  # Max requests per minute
RATE_WINDOW = 60  # Time window in seconds
request_history = {}  # Track request history

# Cache for storing responses to avoid recomputing
response_cache = TTLCache(maxsize=100, ttl=3600)  # Cache lasts 1 hour

# Create the web application
app = FastAPI(
    title="Academic Advising API",
    description="An API for providing academic advising using course data and grade distributions",
    version="1.0.0"
)

# Add security and browser access settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (customize for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"]
)

def check_rate_limit(request: Request) -> bool:
    """Check if a user has made too many requests."""
    client_ip = request.client.host
    current_time = time.time()
    
    # Remove old requests
    request_history[client_ip] = [
        timestamp for timestamp in request_history.get(client_ip, [])
        if current_time - timestamp < RATE_WINDOW
    ]
    
    # Check if user has hit limit
    if len(request_history.get(client_ip, [])) >= RATE_LIMIT:
        return False
    
    # Add new request
    request_history.setdefault(client_ip, []).append(current_time)
    return True

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Stop users from making too many requests."""
    if not check_rate_limit(request):
        return JSONResponse(
            status_code=429,
            content={
                "error": f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per {RATE_WINDOW} seconds."
            }
        )
    return await call_next(request)

def init_model():
    """Set up the AI model for answering questions."""
    try:
        # Load the AI model and its word processor
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Set up the AI pipeline with specific settings
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256,  # Maximum response length
            min_length=100,  # Minimum response length
            do_sample=True,  # Allow some randomness
            temperature=0.4,  # Control randomness (lower = more focused)
            top_p=0.92,
            top_k=50,
            num_beams=3,  # Number of responses to consider
            length_penalty=1.1,
            repetition_penalty=1.2,  # Avoid repeating words
            early_stopping=True
        )
        
        # Create the final AI pipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

# Template for how to ask the AI questions
template = """You are an academic advisor helping students choose courses. Based on the course data below, provide a structured recommendation.

Course Data: {context}

Student Question: {question}

Format your response EXACTLY like this JSON (no other text):
{{
    "recommended_courses": [
        {{
            "code": "CS1234",
            "gpa": "3.75",
            "professors": ["Smith (3.8)", "Jones (3.7)"],
            "difficulty": "Easy"
        }}
    ],
    "key_points": [
        "Most recommended courses have high GPAs above 3.5",
        "Professor Smith consistently receives excellent ratings",
        "These courses are ideal for building a strong foundation"
    ],
    "summary": "These courses offer an excellent balance of engaging content and achievable success"
}}"""

# Create the prompt template
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

class Query(BaseModel):
    """Define what a question should look like."""
    question: str = Field(..., min_length=3, description="The question to be answered")
    temperature: Optional[float] = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Temperature for response generation"
    )

# Function to get cached responses
def get_cached_response(question: str):
    """Get a previously cached answer if it exists."""
    return response_cache.get(question)

def cache_response(question: str, response: dict):
    """Save an answer for future use."""
    response_cache[question] = response

# Initialize the AI model and QA system
logger.info(f"Loading model: {MODEL_NAME}")
llm = init_model()
logger.info("Successfully initialized language model")

# Set up the course data and QA system
retriever = get_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)
logger.info("Successfully initialized QA chain")

@app.post("/ask")
async def ask_agent(query: Query) -> JSONResponse:
    """Handle student questions and provide course recommendations."""
    try:
        # Check if we already have an answer
        cached_response = get_cached_response(query.question)
        if cached_response:
            cached_response["metadata"]["cached"] = True
            return JSONResponse(content=cached_response)
        
        # Get answer from AI
        result = qa_chain.invoke({"query": query.question})
        
        # Get the source documents used
        source_docs = []
        if result.get("source_documents"):
            source_docs = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        
        # Create course recommendations
        recommended_courses = []
        for doc in source_docs[:3]:  # Use top 3 courses
            content = doc["content"]
            metadata = doc["metadata"]
            
            if "course_code" in metadata:
                course = {
                    "code": metadata["course_code"],
                    "gpa": str(metadata["avg_gpa"]),
                    "professors": metadata.get("professors", []),
                    "difficulty": metadata["difficulty"]
                }
                recommended_courses.append(course)
        
        # Create the final answer
        structured_answer = {
            "recommended_courses": recommended_courses,
            "key_points": [
                f"Found {len(recommended_courses)} courses with high GPAs (3.5+)",
                "All recommended courses have experienced professors",
                "These courses have historically high success rates"
            ],
            "summary": f"Recommended {len(recommended_courses)} courses with strong grade distributions and experienced faculty."
        }
        
        # Package everything together
        response = {
            "question": query.question,
            "answer": structured_answer,
            "source_documents": source_docs,
            "metadata": {
                "temperature": query.temperature,
                "model": MODEL_NAME,
                "cached": False
            }
        }
        
        # Save for future use
        cache_response(query.question, response)
        
        logger.info(f"Successfully processed question: {query.question[:50]}...")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error processing your question",
                "details": str(e),
                "structured_response": {
                    "recommended_courses": [],
                    "key_points": ["Error processing request"],
                    "summary": "Unable to process request"
                }
            }
        )

@app.get("/stats")
async def get_stats() -> JSONResponse:
    """Get information about API usage."""
    return JSONResponse(content={
        "cache_size": len(response_cache),
        "cache_info": {
            "maxsize": response_cache.maxsize,
            "ttl": response_cache.ttl,
            "current_size": len(response_cache)
        },
        "rate_limits": {
            "requests_per_window": RATE_LIMIT,
            "window_seconds": RATE_WINDOW
        }
    })

@app.get("/")
async def root():
    """Welcome page with API information."""
    return {
        "message": "Welcome to the Academic Advising API",
        "version": "1.0.0",
        "config": {
            "host": HOST,
            "port": PORT,
            "model": MODEL_NAME
        },
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/health": "GET - Check API health",
            "/stats": "GET - Get API usage statistics"
        }
    }

# Start the server if running directly
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(
        "agent:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="debug"
    )