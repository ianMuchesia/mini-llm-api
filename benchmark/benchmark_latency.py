import requests
import time

url = "http://localhost:8000/generate"
prompt = "Romeo"
token_lengths = [10, 50, 100]

print("Starting Latency Benchmark...\n")

for max_len in token_lengths:
    payload = {
        "prompt": prompt,
        "max_length": max_len,
        "temperature": 0.8
    }
    
    # Record start time
    start_time = time.time()
    
    # Send request to your local Docker container
    response = requests.post(url, json=payload)
    
    # Record end time
    end_time = time.time()
    
    if response.status_code == 200:
        latency = end_time - start_time
        print(f"Max Length: {max_len:3d} tokens | Latency: {latency:.4f} seconds")
    else:
        print(f"Error at {max_len} tokens: HTTP {response.status_code}")