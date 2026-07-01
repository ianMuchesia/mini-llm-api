# Training Notes: Character-Level Transformer LM

## Data Pipeline
- Text is encoded character by character. Each training window: `X = text[i:i+seq_len]`, `Y = text[i+1:i+seq_len+1]`.
- `DataLoader(shuffle=True)` picks random indices — slicing logic must be fully self-contained per sample.
- Built a Wikipedia data pipeline: `extract_swwiki.py` → `clean_wikitext.py` → `swahili_clean.txt`.

## Architecture Decisions
- Learned positional embeddings (`nn.Embedding(max_len, d_model)`) — what GPT actually uses, not sinusoidal.
- Causal mask required for decoder-only behaviour. Without it, the model can see future tokens (BERT behaviour).
- **Silent PyTorch trap:** `nn.TransformerEncoderLayer` defaults `dim_feedforward=2048` regardless of `d_model`. At `d_model=128`, that's 524K params per layer — far too large for small data. Fix: set `dim_feedforward=d_model * 4` explicitly.

## GPU Training
- Always call `model.to(device)` after creating the model. Without it, the model stays on CPU while data goes to GPU — PyTorch won't warn you, it just silently copies every batch back and forth.
- Tesla T4 on Colab: ~5 min/epoch vs ~20 min on CPU.

## Dropout & train/eval mode
- `model.train()` activates dropout (10% of connections randomly zeroed per batch) — forces the network to learn robust patterns.
- `model.eval()` turns dropout off — full network capacity used for validation and generation.
- The `dropout` parameter in `nn.TransformerEncoderLayer` defaults to `0.1`.

## Sampling Strategies

**Greedy:** Always picks the highest logit. Produces coherent early output then locks into repetitive loops (e.g., `"Junadadadadadamii"`). The model is too rigid.

**Temperature:** Divides logits by `T` before softmax. `T < 1` makes it more conservative, `T > 1` makes it more creative. Comes from the Boltzmann distribution in physics — low temp = particles freeze into predictable states, high temp = particles become chaotic.

**Top-k:** Keep only the `k` highest probability tokens, zero out the rest.

**Top-p (nucleus):** Keep the smallest set of tokens whose cumulative probability ≥ `p`. More adaptive than top-k.

## Overfitting vs Underfitting — What I Actually Observed

| Data Size | What Happened |
| :--- | :--- |
| 8KB | Severe overfitting. Model memorized the file. Train 86%, val loss → 2.95 |
| 1MB | Still overfitting. Train/val diverged within 2 epochs |
| 3MB | Overfitting gone. Train ~66%, val ~65%, but **plateaued** — loss not moving |

**Lesson:** It's the ratio between data size and model capacity that matters, not either alone.

## The Plateau Problem (Current)
Training is stuck at ~66% accuracy across epochs 3–6 on 3MB of data. This is **not** overfitting — train and val are close. The model has extracted everything it can at this size.

**Things that fight overfitting:** more data, dropout, weight decay (L2), early stopping.
**Things that break a plateau:** bigger model (`d_model`), lower learning rate, learning rate schedule (cosine annealing).

> **Note:** Adam already includes momentum internally. `weight_decay=1e-5` in the optimizer IS L2 regularization. Cosine annealing is a fancier version of manually dropping the learning rate — same idea, more controlled.

## Why Val Accuracy > Train Accuracy in Epoch 1
Not a bug. Training accuracy is the average over all batches in the epoch, including the early terrible ones. Validation runs once at the very end, after all the weight updates. It's measuring a smarter model than the training average reflects.