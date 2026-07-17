# Inference Latency Report

**Hardware:** Local CPU
**Model:** Mini-LLM (PyTorch)

| Requested Tokens | Latency (Seconds) |
|------------------|-------------------|
| 10               | [0.0124]          |
| 50               | [0.0767]          |
| 100              | [0.1548]          |

**Observations:**
As token count increases, latency scales linearly because autoregressive generation creates one token at a time.