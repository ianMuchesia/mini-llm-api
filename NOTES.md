# Training Notes: Character-Level Transformer LM

## 1. Data Slicing & Loader Indexing
- For autoregressive prediction: Input `X = text[i : i+seq_len]`, Target `Y = text[i+1 : i+seq_len+1]` — at every position the model predicts the next character.
- `DataLoader(shuffle=True)` picks indices randomly, not sequentially. That means index 0, then 1024, then 5 — not 0, 64, 128. Slicing logic in `__getitem__` must be fully self-contained and not assume any ordering.

## 2. Learned Positional Embeddings
- GPT-style architectures use `nn.Embedding(max_len, d_model)` for position — a trainable matrix, not the sinusoidal formula. The sinusoidal approach works too, but PyTorch has no built-in module for it; you'd have to implement the math yourself. GPT-1, GPT-2, GPT-3 all abandoned sinusoidal and used learned embeddings.

## 3. Causal Masking (Decoder-Only Attention)
- Without a causal mask, `nn.TransformerEncoderLayer` attends to all tokens bidirectionally — it can see future tokens, which is BERT behaviour, not GPT.
- Fix: generate an upper-triangular mask and pass it on every forward call:
  ```python
  causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
  out = layer(out, src_mask=causal_mask)
  ```

## 4. Dropout & train/eval Mode
- `model.train()` activates dropout: 10% of connections are randomly zeroed per batch. This forces the network to learn robust patterns rather than memorize exact sequences.
- `model.eval()` turns dropout off — the full network capacity is restored for validation and generation. This is why validation can look slightly better than training: it's running with a "stronger" version of the network.
- The parameter controlling the dropout rate in `nn.TransformerEncoderLayer` is called `dropout`, and its default is `0.1`.

## 5. GPU Training & the `model.to(device)` Trap
- Always call `model.to(device)` immediately after creating the model. If you skip it, the model stays on CPU while the data goes to GPU. PyTorch won't throw an error — it silently copies every batch from GPU to CPU for the forward pass, then sends gradients back. The GPU is completely idle.
- Measured on Colab: ~5 min/epoch on Tesla T4 vs ~20 min/epoch on CPU.
- Confirming GPU is actually being used:
  ```python
  print(torch.cuda.is_available())     # True
  print(torch.cuda.get_device_name(0)) # Tesla T4
  ```
- The `pin_memory=True` warning (`no accelerator is found`) is how you catch this — it was the signal that the session had no GPU, not the code.

## 6. The `dim_feedforward` Silent Default Trap
- `nn.TransformerEncoderLayer` defaults `dim_feedforward=2048` regardless of what `d_model` you pass.
- At `d_model=128`: each layer has two linear projections of size 128→2048 and 2048→128, totalling 2 × 128 × 2048 = **524,288 parameters per layer** — far larger than the embeddings and attention combined.
- This was the root cause of overfitting on 1MB data. The feedforward blocks had way more capacity than the data could support, so they memorized instead of generalizing.
- Fix: explicitly set `dim_feedforward=d_model * 4`:
  ```python
  nn.TransformerEncoderLayer(
      d_model=d_model, nhead=num_heads,
      dim_feedforward=d_model * 4,  # 512 for d_model=128, not 2048
      dropout=0.1, batch_first=True, norm_first=True
  )
  ```

## 7. Greedy Decoding & the Stuttering Trap
- Greedy search always picks the single highest-probability token, 100% of the time.
- The model generated good initial Swahili then collapsed into infinite loops:
  > `Senegal bao katika ushindi wao wa 3-1 dhidi ya Les Bleus Juni 16. 16, Juni Junadadadadadamii Mwa niwa ni nakakakakamba...`
- Why: if the model is 51% sure the next character is "d" and 49% sure it's "m", greedy suppresses "m" every time. That rigidity is what locks it into stuttering loops.

## 8. Temperature Sampling
- The model outputs raw logits (e.g., 8.5, 2.1, -4.0) — not probabilities. To sample, you first convert to probabilities using softmax, then roll a weighted dice with `torch.multinomial`.
- `torch.multinomial(probs, num_samples=1)` returns the **index** of the chosen token, not the token itself. You then decode that index back to a character.
- Any token with probability > 0% can be chosen. Only tokens with exactly 0% probability (those set to `-inf` before softmax) are truly excluded.
- Temperature `T` is a scaling knob on the logits before softmax:
  - `T = 1.0` — standard probabilities
  - `T < 1.0` — exaggerates the gap between scores, model becomes more conservative (closer to greedy)
  - `T > 1.0` — shrinks the gap, model becomes more creative and unpredictable
- The concept comes from the **Boltzmann Distribution** in Statistical Mechanics. At low temperature, particles freeze into predictable low-energy states. At high temperature, they become excited and chaotic. AI researchers borrowed the same formula directly.

## 9. Top-k Sampling
- Keep only the `k` highest-probability tokens and zero out all others (set to `-inf` before softmax).
- The mask logic: sort tokens by score descending, mark everything after rank `k` as True (meaning "remove this"), then apply `-inf` to those positions. Force the #1 token mask to always be `False` so the engine never divides by zero or crashes on an all-masked distribution.
- `k=5` → conservative, `k=50` → more diverse.

## 10. Top-p (Nucleus) Sampling
- Instead of a fixed `k`, keep the smallest set of tokens whose cumulative probability ≥ `p`.
- Sort tokens by probability descending, compute the running cumulative sum, mark everything beyond the threshold as excluded.
- Example: if `p=0.9`, keep tokens until you've accumulated 90% of the total probability mass. The remaining 10% ("garbage predictions") get zeroed out.
- Generally outperforms fixed top-k because the number of kept tokens adapts to the model's confidence — sometimes 3 tokens cover 90%, sometimes 40.

## 11. Why Output Is Gibberish After a Few Sentences
The model generates grammatically correct Swahili early on, then degrades. This isn't a code bug — it's three physical constraints of a mini LLM:

1. **Character-level tokenization:** To write "Shamrock", the model must make 8 perfect consecutive predictions. One mistake (e.g., "Shamrik") derails the entire context chain.
2. **Short context window (`max_len=75`):** Only ~10–15 words of memory. By the time it finishes writing a sentence, the first words have fallen out of the window — the model literally forgets what it was saying.
3. **Small parameter count:** `d_model=128, num_heads=4, num_layers=2` is a few hundred thousand parameters. GPT-3 has 175 billion. There simply isn't enough mathematical depth to sustain long coherent narratives.

## 12. Overfitting vs Underfitting — What the Experiments Showed

| Data Size | Behaviour |
| :--- | :--- |
| 8KB | Severe overfitting: train 86%, val loss climbing to 2.95 |
| 1MB | Still overfitting: train/val diverged within 2 epochs |
| 3MB | Overfitting gone: train ~66%, val ~65%, but **plateau** — loss not moving |

The pattern is clean: as data grew, overfitting shrank. At 3MB the model stopped memorizing, but hit a capacity ceiling instead.

**Rule of thumb (approximate):**
- Less data → more likely to overfit (model memorizes)
- Too few epochs → underfitting (model hasn't learned yet)
- But what actually matters is the **ratio** between data size and model capacity. A huge model on tiny data will always overfit, even with few epochs.

## 13. The Plateau & What To Do About It
Train and val stuck at ~66%/65% across epochs 3–6. Not overfitting — they're moving together. The model has extracted everything it can at this size.

**Things that fight overfitting:** more data, dropout, weight decay (L2 regularization), early stopping.
**Things that break a plateau:** larger `d_model`, lower learning rate, learning rate schedule.

- **Momentum:** already included in Adam internally. You don't need to add it.
- **L2 regularization:** this is `weight_decay=1e-5` already in `optim.Adam(...)`. It restricts model weights from growing too large. Fights overfitting, not plateaus — using it to break a plateau would make things worse.
- **Cosine annealing:** a learning rate schedule that gradually decreases `lr` following a cosine curve. Good for settling into a better minimum after the model gets close. A simpler first test is just manually dropping `lr` from `0.001` to `0.0003`.

## 14. Why Val Accuracy > Train Accuracy in Epoch 1
Not a bug. Training accuracy is the **average over all batches** in the epoch — including the early ones where the model was still randomly initialized and performing terribly. Validation runs **once, at the very end**, after all the weight updates from all 29,300 batches. It's measuring the "best-so-far" model, while training accuracy reflects the whole journey from bad to good. This gap closes from epoch 2 onward. If val loss starts rising while train accuracy keeps climbing, that's the overfitting signal.