from fastapi.testclient import TestClient
from agent import app
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_qa():
    response = client.post(
        "/ask",
        json={"question": "What are the key areas of AI?"}
    )
    logger.debug(f"Response: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data

if __name__ == "__main__":
    print("Testing health endpoint...")
    test_health()
    print("Health check passed!")
    
    print("\nTesting QA endpoint...")
    test_qa()
    print("QA test passed!") 