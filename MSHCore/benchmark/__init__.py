"""Model benchmarking — per-runtime runners and the shared result store.

- ``ollama_runner`` — running prompts against models served by Ollama and
  comparing the timings and noise of the results.
- ``history`` — persisted comparison runs under the toolkit's data root,
  shared by every runner.
"""
