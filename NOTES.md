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
* **Parameter Details**: Controlled by the **`dropout`** argument in `nn.TransformerEncoderLayer`, which defaults to **`0.1`** (a 10% sabotage rate).