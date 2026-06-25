# Mini-LLM: Character Language Model & Sampling

A lightweight PyTorch implementation of a GPT-style character-level Transformer language model, featuring diverse text generation sampling strategies.

---

## 🛠️ Project Structure
* 📂 **`src/`** — Model (`language_model.py`), Dataset (`dataset.py`), Trainer (`trainer.py`), Generation (`generation.py`).
* 📂 **`notebooks/`** — Sampling strategies comparison study (`sampling_comparison.ipynb`).
* 📂 **`experiments/`** — Generated samples under greedy, temperature, top-k, and top-p settings.
* 📂 **`math-notes/`** — Parameter counting and FLOP estimation analyses.
* 📂 **`data/`** — Raw text training corpus (e.g., `shakespeare.txt` or `swahili.txt`).
* 📜 **[NOTES.md](file:///home/msodoki/Desktop/Mathematics/mini-llm-api/NOTES.md)** — Conceptual study guide & revision notes.

---

## 📐 Math to PyTorch Mapping
* **Token & Learned Position Embeddings** $\implies$ `nn.Embedding`
* **Causal Attention Blocks** $\implies$ `nn.TransformerEncoderLayer` with triangular masking
* **Layer Norm & Output Projection** $\implies$ `nn.LayerNorm` & `nn.Linear`

---

## 💻 Quick Start & Generation

### 1. Initialize & Train the Model
```python
from src.language_model import LanguageModel
model = LanguageModel(d_model=128, num_heads=4, num_layers=2, max_len=64, vocab_size=100)
```

### 2. Generate Text with Sampling (TextGenerator)
```python
from src.generation import TextGenerator
generator = TextGenerator(model, tokenizer)
text = generator.generate("ROMEO:", max_new_tokens=100, mode="top_p", p=0.9, temp=0.8)
```
