from fastapi.testclient import TestClient
from app.main import app

# Create a test client that connects to our FastAPI app
client = TestClient(app)

def test_generate_success():
    """Test that a valid payload returns 200 OK and generated text."""
    payload = {
        "prompt": "Romeo",
        "max_length": 10
    }
    response = client.post("/generate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert len(data["generated_text"]) > 0

def test_generate_invalid_input():
    """Test that Pydantic blocks bad data (e.g., empty prompt, too long max_length)."""
    payload = {
        "prompt": "",           # Fails min_length=1
        "max_length": 1000      # Fails le=512
    }
    response = client.post("/generate", json=payload)
    
    # Expect 422 Unprocessable Entity
    assert response.status_code == 422