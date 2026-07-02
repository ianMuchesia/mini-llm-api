# FLOP Estimation (Forward Pass)
**Project:** mini-llm-api (Cycle 8 - Week 3)

## The Computational Logic (The "2N" Rule)
FLOPs (Floating Point Operations) measure the actual hardware math required to process data. When a token passes through a weight matrix, the hardware must execute a Multiply-Accumulate (MAC) operation for every parameter. 

For every single weight, the processor does two things:
1. **Multiplies** the input value by the weight.
2. **Adds** that result to a running total (accumulation).

Because each parameter requires exactly one multiplication and roughly one addition, the standard industry formula to calculate inference compute cost is:
**FLOPs = 2 * Total Parameters * Total Tokens**

---

## Inference Variables
To calculate the cost of a single forward pass (inference) through this specific model architecture, we use the following batch dimensions:
* **Total Parameters (N):** 418,113
* **Batch Size (B):** 32
* **Sequence Length (T):** 64
* **Total Tokens (B * T):** 2,048

## Final Calculation
* **Formula:** 2 * 418,113 * 2,048
* **Result:** 1,712,590,848 Operations

**Total Compute Cost:** ~1.71 GigaFLOPs per forward pass.