# Mini-LLM: Character Language Model & Sampling

A lightweight PyTorch implementation of a GPT-style character-level Transformer language model, featuring diverse text generation sampling strategies.

---

## 🛠️ Project Structure
* 📂 **`src/`** — Model (`language_model.py`), Dataset (`dataset.py`), Trainer (`trainer.py` with early stopping), Generation (`generation.py`).
* 📂 **`experiments/`** — Training history (`training_data.json`), greedy sample output (`greedy_examples.txt`), and temperature samples (`temperature_examples.txt`).
* 📂 **`notebooks/`** — Sampling strategies comparison study (`sampling_comparison.ipynb`).
* 📂 **`math-notes/`** — Parameter counting and FLOP estimation analyses.
* 📂 **`data/`** — Raw text training corpus (e.g., `shakespeare.txt` or `swahili.txt`).
* 📜 **[NOTES.md](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/NOTES.md)** — Conceptual study guide & revision notes.

---

## 📐 Math to PyTorch Mapping
* **Token & Learned Position Embeddings** $\implies$ `nn.Embedding`
* **Causal Attention Blocks** $\implies$ `nn.TransformerEncoderLayer` with triangular masking
* **Layer Norm & Output Projection** $\implies$ `nn.LayerNorm` & `nn.Linear`
* **Temperature Adjustment** $\implies$ $p_i = \text{softmax}(\text{logits} / T)$ sampled via `torch.multinomial`

---

## 💻 Quick Start & Generation

### 1. Initialize & Train the Model (with Early Stopping)
```python
from src.language_model import LanguageModel
model = LanguageModel(d_model=128, num_heads=4, num_layers=2, max_len=64, vocab_size=100)
# Model checkpoint will be saved to checkpoints/best_model.pt
```

### 2. Generate Text with Temperature Sampling
```python
# Greedy Generation (Default)
out = model.generate(x, max_new_chars=200)

# Temperature Sampling (e.g., T = 0.8 for creative variety)
out = model.generate(x, max_new_chars=200, temperature=0.8)
```
