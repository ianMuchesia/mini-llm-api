# mini-llm-api

A GPT-style character-level language model trained on Swahili Wikipedia, with a FastAPI streaming inference server. Built over 3 weeks as a ground-up study of transformers, sampling strategies, and MLOps deployment.

---

## What's Inside

| Folder | Contents |
| :--- | :--- |
| `src/` | Model, tokenizer, dataset, trainer, generation scripts, Wikipedia pipeline |
| `app/` | FastAPI server with `/generate` and `/generate_stream` endpoints |
| `experiments/` | Sampling outputs from CPU and GPU runs |
| `math-notes/` | Parameter count analysis (~418K params) and FLOP estimation (~1.71 GFLOPs) |
| `checkpoints/` | Saved model weights (`best_model.pt`) and vocab |

---

## Model Architecture
- **Type:** Decoder-only Transformer (GPT-style)
- **Tokenization:** Character-level (`CharTokenizer`)
- **Embeddings:** Learned token + learned positional (`nn.Embedding`)
- **Attention:** Causal (upper-triangular mask), `nn.TransformerEncoderLayer`
- **Config:** `d_model=128`, `num_heads=4`, `num_layers=2`, `max_len=75`
- **Training data:** Swahili Wikipedia (~5MB clean text)

## Sampling Strategies Implemented
- **Greedy** — always picks the top token (fast, but stutters)
- **Temperature** — scales logits before softmax; `T<1` = conservative, `T>1` = creative
- **Top-k** — keeps only the `k` highest-probability tokens
- **Top-p (nucleus)** — keeps the smallest set of tokens covering cumulative probability ≥ `p`

---

## API Endpoints

```
POST /generate         → returns full generated text
POST /generate_stream  → streams tokens one-by-one (Server-Sent Events)
GET  /health           → health check
```

**Request body:**
```json
{ "prompt": "Habari za", "max_length": 200, "temperature": 0.8, "top_k": 40 }
```

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn app.main:app --reload --port 8000
```

```bash
# Docker (development)
docker-compose up
```

---

## Study Notes
See [NOTES.md](NOTES.md) for a full conceptual breakdown of everything learned: causal masking, sampling math, overfitting experiments, Docker patterns, and streaming generation.
