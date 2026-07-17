# Inference Throughput Report

**API Framework:** FastAPI + Uvicorn
**Deployment:** Docker Container (CPU)

## Single Request Baseline
* **Time taken:** [0.0287]
* **Throughput:** [34.90] requests/second

## Concurrent Stress Test
* **Total Requests:** 20
* **Concurrent Users:** 4
* **Total Time:** [0.4659]
* **Throughput:** [42.93] requests/second

**Observations:**
Uvicorn handles concurrent connections well, but overall RPS(Request per second) is bottlenecked by the sequential nature of PyTorch CPU matrix multiplications.