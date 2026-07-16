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

You got it. Let’s break down the exact mechanics of what is happening under the hood when you use both `COPY` in your `Dockerfile` and `volumes` in your `docker-compose.yml`.

This is a core MLOps concept called **environment parity** — maintaining the exact same architecture for both local development and production, but optimizing how files are handled in each state.

Here is the deep dive into how these two instructions interact and why you engineer it this way.

## 1. The `Dockerfile` (`COPY`): The Immutable Production Artifact

When you run `docker build`, Docker executes the `COPY` instructions step-by-step. It physically duplicates your host files and bakes them into read-only layers inside the Docker image.

* **The Goal:** True portability.
* **The Result:** You can push this image to Docker Hub, an AWS Elastic Container Registry, or a colleague's laptop. When they run `docker run mini-llm-api`, the API spins up immediately. They do not need to clone your Git repo, and they do not need to download the `best_model.pt` file separately. The image contains 100% of the DNA needed to run.

**The ML Catch:** Model checkpoints are heavy. If `best_model.pt` is 1.5GB, your resulting Docker image will be at least 1.5GB larger. Pushing and pulling this image across the internet takes time and costs bandwidth.

## 2. The `docker-compose.yml` (`volumes`): The Local Development Override

When you spin up the container using Docker Compose, the `volumes` directive essentially punches a hole through the container’s isolated filesystem and connects it directly to your host machine's hard drive.

* **The Shadowing Effect:** Because you mounted `./src:/app/src`, the container **completely ignores** the `src/` folder that was baked into it during the `COPY` step. Instead, it looks directly at the `src/` folder on your local machine.
* **The Developer Experience:** If you find a bug in `generation.py` and fix it in VS Code, you just hit save. Because Uvicorn is running with the `--reload` flag, it detects the file change through the volume mount and restarts the API instantly.

If you *only* relied on the `COPY` command without volumes, you would have to stop the container, run a 2-minute `docker build` to bake the new code into a new image, and run it again. Volumes save you hours of waiting over a development cycle.

## 3. Why We Comment Out Volumes in Production

In your previous projects, you likely commented out the `volumes` section (or used a separate `docker-compose.prod.yml`) when deploying to a live server. Here is why:

If you deploy to an AWS EC2 instance and leave `volumes: - ./src:/app/src` active, the container will look for a local `./src` folder on the AWS server. But because you only deployed the Docker image (and didn't manually transfer your source code files to the EC2 hard drive), that folder doesn't exist on the host. The container will crash or run empty.

## The Strategy Summary

By writing your files the way you did, you achieved the perfect hybrid:

1. **You build with `COPY**` so the image is theoretically ready to be shipped anywhere.
2. **You run locally with `volumes**` so you can tweak your code and swap model checkpoints dynamically without rebuilding a 2GB image every time.

---

Since your `Dockerfile` and Compose files are structurally sound, the next step in the Week 4 syllabus is to actually build the image and see how big it is with the PyTorch CPU wheels and your checkpoint inside.

Are you ready to run the `docker build -t mini-llm-api .` command in your terminal, or do you need to create a `.dockerignore` file first to prevent Docker from copying unwanted junk (like `__pycache__` or virtual environments) into the image?