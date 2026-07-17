FROM python:3.11-slim

WORKDIR /app

# Step 1: Install CPU PyTorch FIRST. 
# Because this is on its own line, Docker caches it permanently.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Step 2: Copy and install the rest of the requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY app/ ./app/


# COPY checkpoints/ ./checkpoints/

# COPY experiments/ ./experiments/

# COPY src/ ./src/


EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]