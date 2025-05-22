from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(query: Query):
    # Placeholder logic
    return {"answer": f"Based on your input: '{query.question}', here are your class options..."}
