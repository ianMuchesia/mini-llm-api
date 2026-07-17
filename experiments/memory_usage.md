# Memory Footprint Report

## Disk Storage
* **Docker Image Size:** [7.68GB / Target: 1.5GB]
* **Model Checkpoint Size:** [8.1MB]

## RAM Usage (Containerized)
* **Idle (Cold Start Loaded):** [386MB] 
* **Peak Inference Usage:** []

**Observations:**
The base FastAPI application requires minimal RAM. The vast majority of the memory footprint comes from loading the model weights into memory at startup. Peak inference adds a slight memory overhead to hold the generated context window and intermediate tensor states.