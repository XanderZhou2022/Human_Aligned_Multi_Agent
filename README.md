# Financial QA Retrieval & Reranking (AI4F '25)

This repository contains the reference implementation accompanying our paper submitted to the ACM AI4F '25 workshop:


The code implements a human-aligned retrieval and zero-shot reranking pipeline for financial question answering over long corporate reports. It mirrors an analyst-style workflow: evidence capture as structured "cards", question intent parsing, staged retrieval, and finance-aware tie-breaking.

## Repository layout

- `finance_rerank_core.py`: Core pipeline for retrieval and LLM-assisted reranking, plus batch evaluation utilities.
- `query_core.py`: Query preprocessing utilities to infer topic, entities, metrics, time filters, and numeric intent.
- `cards.py`: Data structures/utilities for compact evidence "cards" (entity, metric, period, numbers, summary/span).

Note: Files were distilled from notebooks; some paths default to Google Colab (`/content/drive/...`). Update paths to your local environment as needed.

## Environment

Recommended Python: 3.9–3.11

Install dependencies (pin as appropriate for your environment):

```bash
pip install pandas numpy tqdm matplotlib openai
```

For LLM reranking, set your API key via environment variables instead of hardcoding in source:

```bash
export OPENAI_API_KEY=your_key
```

Then modify the client initialization in `finance_rerank_core.py` to read from the environment (or provide your own client wrapper).

## Quick start

1) Prepare inputs
- Cards dataset: a JSONL where each line is a card with fields like `summary`, `entities`, `metrics`, `numbers`, `chunk_uid`, `topic`, `section`, `period_norm`, etc.
- Question index: a DataFrame with columns like `uid_question`, `question`, and parsed `ques_result`.
- Sample mapping: `sample_id → {chunk_index → chunk_uid}` for document-local chunk access.

2) Configure paths
- In `finance_rerank_core.py`, replace the example `path` variables that reference `/content/drive/...` with your local file paths.

3) Run a sample rerank
- Use `get_top_chunks_for_sample_llm(sample_id=..., k=10)` to obtain top-k chunk indices per sample.
- Use `batch_run(...)` to process multiple samples and write CSV/JSONL outputs.

Example (conceptual):
```python
from finance_rerank_core import get_top_chunks_for_sample_llm

# Ensure globals like cards_df, sample_to_chunks, sample_to_uidq, questionid_to_details are loaded
top10 = get_top_chunks_for_sample_llm(sample_id=123, k=10)
print(top10)
```

4) Metrics
- The script includes utilities to compute `ndcg@5`, `map@5`, and `mrr@5` against label1/label2 ground truth.

## Notes on the CLI

A minimal CLI wrapper is present at the bottom of `finance_rerank_core.py` as a placeholder. It prints a warning unless a `run_pipeline(...)` function is defined in your environment. Most users should import the module and call the functions directly, as shown above.



## License

`query_core.py` includes an SPDX header for Apache-2.0. Unless specified otherwise, we release this repository under the Apache License 2.0.

```text
SPDX-License-Identifier: Apache-2.0
```

## Disclaimer

- This implementation is provided for research purposes. You are responsible for compliance with your data, model, and API providers’ terms and any regulatory constraints relevant to your use case.
- Paths and lightweight data loaders are examples; adapt to your corpus and storage format.

