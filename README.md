# Mini-LLM API & Character Language Model

A lightweight, PyTorch character-level GPT-style Transformer language model, built to be containerized and exposed via a FastAPI REST API.

---

## 🛠️ Project Structure
* 📂 **`src/`** — Core PyTorch modules.
  * [dataset.py](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/dataset.py) — Sliding window token selection.
  * [tokenizer.py](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/tokenizer.py) — Character vocabulary encoder/decoder.
  * [language_model.py](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/src/language_model.py) — GPT-style Transformer.
* 📂 **`data/`** — Training corpus (e.g., [swahili.txt](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/data/swahili.txt)).
* 📜 **[NOTES.md](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/NOTES.md)** — Conceptual study guide & revision notes.

---

## 📐 Math to PyTorch Mapping
* **Token & Position Embeddings** $\implies$ `nn.Embedding`
* **Attention Blocks** $\implies$ `nn.TransformerEncoderLayer` (with custom causal masks)
* **Layer Norm & Output Head** $\implies$ `nn.LayerNorm` & `nn.Linear`

---

## 💻 Quick Start & API Mockup

### 1. Initialize the Model
```python
from src.language_model import LanguageModel
model = LanguageModel(d_model=256, num_heads=8, num_layers=4, max_len=64, vocab_size=100)
```

### 2. Run API via Docker (Upcoming)
```bash
docker build -t mini-llm-api .
docker run -p 8000:8000 mini-llm-api
curl -X POST "http://localhost:8000/generate" -d '{"prompt": "Morocco"}'
```

---

## 🧭 Roadmap
* [x] Core GPT Architecture & Tokenizer
* [ ] FastAPI/Flask REST API wrapper
* [ ] Docker containerization & deployment
