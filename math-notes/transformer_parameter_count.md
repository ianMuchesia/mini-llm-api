# Transformer Parameter Count Analysis
**Project:** mini-llm-api (Cycle 8 - Week 3)

## Model Configuration
* **Vocabulary Size:** 65 (Tiny Shakespeare)
* **Embedding Dimension (d_model):** 128
* **Context Length:** 64
* **Attention Heads:** 4
* **Transformer Layers:** 2

---

## 1. Embedding Layers
The embedding layers act as the initial lookup tables, translating raw token IDs and their positions into dense vectors.
* **Token Embeddings:** 65 (Vocab) * 128 (d_model) = 8,320 parameters
* **Positional Embeddings:** 64 (Context) * 128 (d_model) = 8,192 parameters
* **Embedding Total:** 16,512 parameters

## 2. Transformer Blocks
The core "brain" of the model consists of 2 identical layers. Each layer contains a Multi-Head Attention mechanism and a Feed-Forward Network (FFN).

### Multi-Head Attention (per layer)
Standard attention uses 4 separate weight matrices (Query, Key, Value, and Output projection). Each maps the 128-dimensional input to a 128-dimensional output space.
* **Single Matrix:** 128 * 128 = 16,384 parameters
* **All 4 Matrices:** 16,384 * 4 = 65,536 parameters

### Feed-Forward Network (per layer)
The FFN expands the model's dimension by a factor of 4 for processing, then projects it back down. 
* **Expansion Layer:** 128 * 512 = 65,536 parameters
* **Projection Layer:** 512 * 128 = 65,536 parameters
* **FFN Total:** 131,072 parameters

* **Total for ONE Block:** 65,536 (Attention) + 131,072 (FFN) = 196,608 parameters
* **Total for TWO Blocks:** 196,608 * 2 = 393,216 parameters

## 3. Final Linear Projection
The final layer maps the model's internal 128-dimensional thoughts back into the 65 possible characters of the vocabulary to generate logits. This layer includes a bias term for each output character.
* **Linear Layer:** (128 * 65) + 65 = 8,385 parameters

---

## Absolute Total Parameter Count
16,512 (Embeddings) + 393,216 (Transformer Blocks) + 8,385 (Linear Output)
**Total = 418,113 parameters**