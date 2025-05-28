# Academic Advisor AI

An AI-powered academic advising system that helps students choose courses based on grade distributions and professor information.

## Features

- Course recommendations based on AI analysis
- Grade distribution insights
- Professor recommendations
- Difficulty assessments
- Real-time chat interface

## Tech Stack

- Backend: FastAPI + LangChain + Hugging Face models
- Frontend: HTML + TailwindCSS
- Deployment: Vercel

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
uvicorn api.index:app --reload --port 3000
```

3. The frontend will be served automatically with the API.

## Environment Variables

Required environment variables:
- `PYTHONPATH`: Set to "."

## Deployment

This project is configured for deployment on Vercel. Simply push to the main branch to trigger a deployment. 