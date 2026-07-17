from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_generate_stream_success():
    """Test that the streaming endpoint returns successfully and yields data."""
    payload = {
        "prompt": "Romeo",
        "max_length": 5
    }
    
    # TestClient buffers the stream for us to inspect
    response = client.post("/generate_stream", json=payload)
    
    assert response.status_code == 200
    # The response text should not be empty (it will contain the generated tokens)
    assert len(response.text) > 0