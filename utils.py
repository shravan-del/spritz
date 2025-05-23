"""
This file handles all the data processing for our academic advising system.
It loads course data, processes it, and prepares it for the AI to use.
"""

from typing import List, Optional, Dict
from pathlib import Path
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import os
import json
from dotenv import load_dotenv

# Load environment settings
load_dotenv()

# Set up logging to track errors and info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File locations and settings
DATA_DIR = Path("data")  # Where we store our data files
CHUNK_SIZE = 300  # How much text to process at once
CHUNK_OVERLAP = 30  # How much text should overlap between chunks
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Model for understanding text

def load_json_file(file_path: Path) -> List[Document]:
    """
    Read a JSON file and turn it into a format our system can use.
    Different files have different formats, so we handle each type differently.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if "vt_courses" in file_path.name:
            # Handle course list file (simple list of course names)
            return [
                Document(
                    page_content=course,
                    metadata={"source": file_path.name, "type": "course"}
                )
                for course in data
            ]
        elif "reddit" in file_path.name:
            # Handle Reddit discussions about courses
            loader = JSONLoader(
                file_path=str(file_path),
                jq_schema=".[]",
                content_key="text",
                text_content=False
            )
            return loader.load()
        else:
            # Handle grade distribution data
            loader = JSONLoader(
                file_path=str(file_path),
                jq_schema=".[]",
                content_key="GPA",
                text_content=False
            )
            return loader.load()
            
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}")
        raise

def get_documents() -> List[Document]:
    """
    Load all our course data files and combine them.
    This includes course lists, grade data, and student discussions.
    """
    files = ["anaanu_data.json", "reddit_data.json", "vt_courses.json"]
    docs = []
    
    for file in files:
        file_path = DATA_DIR / file
        if not file_path.exists():
            logger.error(f"Required data file not found: {file}")
            raise FileNotFoundError(f"Missing data file: {file}")
            
        try:
            file_docs = load_json_file(file_path)
            logger.info(f"Successfully loaded {len(file_docs)} documents from {file}")
            docs.extend(file_docs)
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
            raise
    
    return docs

def load_course_data() -> List[Dict]:
    """
    Load and combine course information with grade data.
    This gives us a complete picture of each course.
    """
    # Load the basic course list
    with open('data/vt_courses.json', 'r') as f:
        courses = json.load(f)
    
    # Load the grade distribution data
    with open('data/anaanu_data.json', 'r') as f:
        grade_data = json.load(f)
        
    # Combine course info with grade data
    course_info = {}
    for course in courses:
        course_info[course] = {
            'code': course,
            'grade_data': []
        }
    
    for entry in grade_data:
        if entry['course'] in course_info:
            course_info[entry['course']]['grade_data'].append({
                'professor': entry['classes_taught'],
                'a_percentage': entry['A_grade_percentage'],
                'gpa': entry['GPA']
            })
            
    return list(course_info.values())

def create_course_documents(course_data: List[Dict]) -> List[Document]:
    """
    Turn our course data into a format that's easy to search and understand.
    We calculate statistics and organize information about each course.
    """
    documents = []
    
    for course in course_data:
        if course['grade_data']:
            # Calculate average grades and success rates
            avg_gpa = sum(float(g['gpa']) for g in course['grade_data']) / len(course['grade_data'])
            avg_a = sum(float(g['a_percentage']) for g in course['grade_data']) / len(course['grade_data'])
            
            # Find the best professors based on GPA
            unique_profs = {}
            for entry in course['grade_data']:
                prof = entry['professor'].strip()
                gpa = float(entry['gpa'])
                if prof not in unique_profs or gpa > unique_profs[prof]['gpa']:
                    unique_profs[prof] = {'gpa': gpa}
            
            # Sort professors by their GPAs
            sorted_profs = sorted(
                [(prof, data) for prof, data in unique_profs.items()],
                key=lambda x: x[1]['gpa'],
                reverse=True
            )[:2]  # Keep only top 2 professors
            
            # Create organized course information
            content = {
                "course_code": course['code'],
                "average_gpa": round(avg_gpa, 2),
                "a_percentage": round(avg_a, 1),
                "professors": [f"{p[0]} ({p[1]['gpa']:.2f})" for p in sorted_profs],
                "difficulty": "Easy" if avg_gpa >= 3.5 else "Moderate" if avg_gpa >= 3.0 else "Challenging"
            }
            
            # Format the information as text
            content_str = (
                f"{content['course_code']} | "
                f"GPA: {content['average_gpa']} | "
                f"A%: {content['a_percentage']} | "
                f"Professors: {', '.join(content['professors'])} | "
                f"Difficulty: {content['difficulty']}"
            )
            
            # Create a searchable document with all the information
            doc = Document(
                page_content=content_str,
                metadata={
                    'course_code': content['course_code'],
                    'avg_gpa': content['average_gpa'],
                    'difficulty': content['difficulty'],
                    'professors': content['professors'],
                    'a_percentage': content['a_percentage']
                }
            )
            documents.append(doc)
    
    return documents

def get_retriever(k: int = 4):
    """
    Create a system that can quickly find relevant courses based on questions.
    This helps the AI give good course recommendations.
    """
    # Load and organize all course data
    course_data = load_course_data()
    documents = create_course_documents(course_data)
    
    # Break documents into searchable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        separators=[" | ", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Set up the system that understands course descriptions
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create a searchable database of courses
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    # Return a system that can find relevant courses
    return vectorstore.as_retriever(
        search_type="mmr",  # Use a method that finds diverse results
        search_kwargs={
            "k": k,  # How many courses to return
            "fetch_k": k * 2,  # How many to consider
            "lambda_mult": 0.7  # Balance between relevance and diversity
        }
    )
