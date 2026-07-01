# Mini-LLM: Character Language Model & Sampling

A lightweight PyTorch GPT-style character-level Transformer, with greedy, temperature, top-k, and top-p text generation sampling strategies.

---

## 🛠️ Project Structure
* 📂 **`src/`** — Model (`language_model.py`), Dataset (`dataset.py`), Trainer (`trainer.py`), Generation (`generation.py`), Wikipedia pipeline (`harvest_sw.py`, `extract_swwiki.py`, `clean_wikitext.py`).
* 📂 **`experiments/`** — Sampling outputs: `greedy_examples.txt`, `temperature_examples.txt`, `topk_examples.txt`, `topp_examples.txt`.
* 📂 **`math-notes/`** — Parameter counting and FLOP estimation analyses.
* 📂 **`data/`** — Training corpus (Swahili Wikipedia clean text).
* 📜 **[NOTES.md](NOTES.md)** — Conceptual study guide & experiment log.

---

## 📐 Math to PyTorch Mapping
* **Token & Learned Position Embeddings** → `nn.Embedding`
* **Causal Attention** → `nn.TransformerEncoderLayer` + upper-triangular mask
* **Temperature Sampling** → $p_i = \text{softmax}(\text{logits} / T)$ via `torch.multinomial`
* **Top-p (Nucleus)** → sort by probability, cumulative sum ≥ p, zero out the rest

---

## 💻 Quick Start

### Train
```python
from src.language_model import LanguageModel
model = LanguageModel(d_model=128, num_heads=4, num_layers=2, max_len=75, vocab_size=100)
model = model.to(device)  # critical — model must be on GPU
```

### Generate
```python
# Greedy
out = model.generate(x, max_new_chars=200)

# Temperature
out = model.generate(x, max_new_chars=200, temperature=0.8)
```

### Run
```bash
python src/trainer.py    # trains with early stopping, saves best_model.pt
python src/generation.py # runs all 4 sampling strategies
```
