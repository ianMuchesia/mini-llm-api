# Revision Notes: Character Transformer LM

## 1. Data Slicing & Loader Indexing
* **Target Mapping:** Input $X_t \implies$ Target $Y_t = X_{t+1}$ via window offsets.
* **Dataloader Shuffling:** With `shuffle=True`, PyTorch samples indices randomly (not sequentially). Slicing logic must be self-contained and index-independent.

## 2. Learned Positional Embeddings
* Modern GPT architectures use trainable learned positional embeddings via `nn.Embedding(max_len, d_model)` instead of fixed sinusoidal features.

## 3. Causal Masking (Decoder-Only Attention)
* Without a causal mask, PyTorch's `nn.TransformerEncoderLayer` allows future tokens to leak (bidirectional attention).
* **Fix:** Generate an upper-triangular mask:
  ```python
  causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
  out = layer(out, src_mask=causal_mask)
  ```

## 4. Dropout (Network Sabotage)
* **`model.train()`**: Randomly deactivates activations during each batch to prevent overfitting/memorization.
* **`model.eval()`**: Deactivates the dropout layer so the model runs at full capability for validation and generation.
* **Parameter Details**: Controlled by the **`dropout`** argument in `nn.TransformerEncoderLayer` (default: `0.1`, representing a 10% dropout rate).

---

## 5. Greedy Decoding & The Stuttering Trap (Day 4)
* **Greedy Search:** Always picks the token with the highest logit probability ($100\%$ confidence selection).
* **The Repetitive Loop Flaw:** In your experiments with Swahili text generation, the model got stuck in an infinite, stuttering loop:
  > *Example output:* `Senegal bao katika ushindi wao wa 3-1 dhidi ya Les Bleus Juni 16. 16, Juni Junadadadadadamii Mwa niwa ni nakakakakamba tishicha 26: yara...`
* **Why it happens:** Real language has uncertainty. If the model is 51% sure the next character is "d" and 49% sure it is "m", greedy decoding completely suppresses "m" every time, leading to rigid, robotic repeating patterns.

---

## 6. The Temperature Fix 🎲 & Physics Analogy (Day 5)
* **The Fix:** Convert raw logits (which are unbounded, like 8.5, 2.1, or -4.0) into valid probability distributions (summing to 1.0) using the **Softmax** function, then sample using probabilities (`torch.multinomial`).
* **Temperature Adjustment ($T$):** A scaling factor applied to raw logits:
  $$p_i = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$
  * **$T = 1.0$ (Standard)**: Standard probabilities derived from the model's logits.
  * **$T < 1.0$ (Cold)**: Exaggerates the differences between scores. Pushes the model closer to deterministic greedy search.
  * **$T > 1.0$ (Hot)**: Shrinks the gap between logits. Increases randomness and creativity, giving "underdog" tokens a higher chance of selection.

### 🌡️ The Boltzmann Distribution Connection
This concept was not a guess—it is adapted directly from **Statistical Mechanics**:
* In physics, the **Boltzmann Distribution** describes particle states.
* **Low Temperature:** Particles freeze and lock into the lowest, most predictable energy state.
* **High Temperature:** Particles get excited, move around randomly, and exist in many different, less probable states.
* By treating raw model output logits as particle energy states, AI researchers copy this thermodynamic equation to control text diversity.