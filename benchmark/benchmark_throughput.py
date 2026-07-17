import requests
import time
import concurrent.futures

URL = "http://localhost:8000/generate"
PAYLOAD = {"prompt": "Romeo", "max_length": 20} # Keep tokens low to focus on API handling

# Test settings
CONCURRENT_USERS = 4
TOTAL_REQUESTS = 20

def send_request():
    response = requests.post(URL, json=PAYLOAD)
    return response.status_code

print("--- Throughput Benchmark ---")

# 1. Single Request Baseline
print("Testing single request throughput...")
start_single = time.time()
send_request()
time_single = time.time() - start_single

print(f"Single Request Time: {time_single:.4f} seconds")
print(f"Single Request Throughput: {1 / time_single:.2f} requests/second\n")

# 2. Multiple Concurrent Requests
print(f"Sending {TOTAL_REQUESTS} requests ({CONCURRENT_USERS} users at a time)...")
start_multi = time.time()

# Spin up multiple threads to hit the API concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
    futures = [executor.submit(send_request) for _ in range(TOTAL_REQUESTS)]
    concurrent.futures.wait(futures)

time_multi = time.time() - start_multi
rps = TOTAL_REQUESTS / time_multi

print(f"Total Time for {TOTAL_REQUESTS} requests: {time_multi:.4f} seconds")
print(f"Concurrent Throughput: {rps:.2f} requests/second")