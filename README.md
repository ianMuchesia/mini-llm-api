# Mini-LLM API & Character Language Model

A lightweight, PyTorch-based character-level GPT-style Transformer language model, designed for conceptual study and text generation, and built to be containerized and exposed via a REST API.

---

## 🧭 Project Roadmap & Core Goals

This project serves as a conceptual and hands-on guide to building and deploying small-scale language models. 
* [x] **Core Architecture**: PyTorch implementation of a GPT-style decoder-only transformer with learned positional embeddings.
* [x] **Data Pipeline**: Character-level tokenization and overlap-sliced dataset creation.
* [ ] **FastAPI/Flask API**: A REST API layer to handle text generation requests dynamically (serving model inferences).
* [ ] **Dockerization**: Containerizing the model environment and API server for standard, cross-platform deployment.
* [ ] **Web Interface**: A minimal, beautiful frontend to interact with the model in real-time.

---

## 🛠️ Project Structure

Below is the layout of the repository mapping files to their functional roles:

* 📂 **`src/`** — Core PyTorch modules and training logic.
  * [dataset.py](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/dataset.py) — Handles sliding window token selection and PyTorch dataset loader integration.
  * [tokenizer.py](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/tokenizer.py) — Character-level vocabulary builder, encoder, and decoder.
  * [language_model.py](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/language_model.py) — Decoder-only GPT Transformer architecture.
* 📂 **`notebooks/`** — Interactive notebooks for visual analysis, tracking training loss, and rendering attention heatmaps.
* 📂 **`math-notes/`** — Detailed study notes on positional encodings, self-attention, and layer normalizations.
* 📂 **`data/`** — Raw text assets for model training (e.g., Swahili dataset `swahili.txt`).
* 📜 **[NOTES.md](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/NOTES.md)** — High-yield conceptual revision notes and silent traps guide.
* 📜 **`Dockerfile`** *(Upcoming)* — Build configuration for containerizing the API server.

---

## 📐 Mathematical to PyTorch Mapping

To make the architecture clear, here is how the mathematical concepts map directly to our code implementation:

| Mathematical Concept | Formulations / Notes | PyTorch Component |
| :--- | :--- | :--- |
| **Token Embedding** | $X_{\text{emb}} = \text{Embed}(X)$ | [nn.Embedding](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/language_model.py#L14) |
| **Positional Embedding** | $P \in \mathbb{R}^{T \times D_{\text{model}}}$ (Learned) | [nn.Embedding](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/language_model.py#L16) |
| **Attention Layers** | $\text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right)V$ (Causal $M$) | [nn.TransformerEncoderLayer](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/language_model.py#L18) |
| **Layer Normalization** | $\text{LN}(x)$ before Attention & FFN | [nn.LayerNorm](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/language_model.py#L25) |
| **Output Logits** | $\hat{y} = X_{\text{final}} W_{\text{head}}$ | [nn.Linear](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/language_model.py#L23) |

---

## 💻 Quick Start & Usage

### 1. Model Initialization
Initialize the Transformer model in Python:

```python
import torch
from src.language_model import LanguageModel

# Model configuration
model = LanguageModel(
    d_model=256,
    num_heads=8,
    num_layers=4,
    max_len=64,
    vocab_size=100
)

# Dummy inputs [Batch Size, Seq Len]
x = torch.randint(0, 100, (8, 64))
logits = model(x)
print("Output logits shape:", logits.shape)  # Expected: [8, 64, 100]
```

### 2. API Usage (Mockup Preview)
Once the API layer is completed, you can spin up the service and generate text with a curl request:

```bash
# Start the Docker container
docker build -t mini-llm-api .
docker run -p 8000:8000 mini-llm-api

# Query the model for predictions
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Morocco iliweka historia", "max_tokens": 50}'
```

---

## 📈 Experimental Results

*(To be populated as training runs are executed)*

| Configuration | Parameters | Epochs | Final Training Loss | Generalization Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline CharTransformer** | `d=256, l=4, h=8` | TBD | TBD | TBD |
