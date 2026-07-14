# Changelog

This file records project iterations so regressions can be traced back to the likely change that introduced them.

## 2026-07-14

### Added

- Added four course PDFs to the local index through `manifest.yaml`: Image Processing, Reinforcement Learning, Deep Reinforcement Learning, and Recurrent Networks.
- Expanded the fixed retrieval evaluation set from 20 to 42 cases.
- Added explicit source/page relevance judgments for multi-source cases.
- Added Hit@k, P95 retrieval latency, and ignored JSON evaluation reports.
- Added a Streamlit retrieval-mode selector.

### Changed

- Added lightweight English plural normalization to BM25 tokenization.
- Cleaned retrieval noise such as the course identifier from English queries while preserving the original question for generation.
- Increased the local index from 9 PDFs / 216 chunks to 13 PDFs / 356 chunks.

### Verified

```text
24 tests passed
rerank @ 6, 42 cases:
source_recall=1.00, page_recall=1.00, MRR=0.964, nDCG=0.861
```

## 2026-07-13

### Added

- Added BM25 keyword retrieval using `rank-bm25`.
- Added hybrid retrieval with Reciprocal Rank Fusion (RRF).
- Added configurable `vector`, `bm25`, and `hybrid` retrieval modes.
- Added `--retrieval-mode` overrides to the `ask` and `eval-retrieval` commands.
- Added retrieval source and fusion score fields to CLI, Streamlit, and query logs.
- Added BM25 exact-term, RRF fusion, and retrieval-mode validation tests.

### Verified

- Unit tests:
  ```text
  16 passed
  ```
- Existing five-case retrieval smoke evaluation at `top_k=6`:
  ```text
  vector: source_recall=1.00, page_recall=1.00
  bm25:   source_recall=1.00, page_recall=1.00
  hybrid: source_recall=1.00, page_recall=1.00
  ```
- These results verify no regression on the existing cases; they do not establish a hybrid-search improvement because the evaluation set is still small.

### Retrieval Evaluation And Reranking

- Expanded the fixed retrieval evaluation set from 5 to 20 cases.
- Added MRR@k and binary nDCG@k ranking metrics.
- Added fixed retrieval queries to non-English evaluation cases for reproducible ablations.
- Added optional cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Added explicit fallback to RRF order when the reranker is unavailable.
- Changed the default retrieval mode to `rerank` after the fixed evaluation showed a ranking improvement.

```text
vector: source/page recall=1.00, MRR=0.960, nDCG=0.870, avg=53ms
bm25: source/page recall=1.00, MRR=0.863, nDCG=0.796, avg=16ms
hybrid: source/page recall=1.00, MRR=0.938, nDCG=0.866, avg=55ms
rerank: source/page recall=1.00, MRR=0.975, nDCG=0.900, avg=530ms
```

Latency numbers are from one local CPU run and are used only for relative comparison within this project.

## 2026-06-22

### Added

- Added project documentation under `docs/`.
  - `USER_GUIDE.md`: local usage, indexing, querying, Streamlit, logs, and troubleshooting.
  - `DEVELOPMENT_PROCESS.md`: step-by-step project build process.
  - `CODE_READING_GUIDE.md`: recommended source-code reading order.
  - `RESUME_PROJECT.md`: Chinese resume-ready project description.
- Verified incremental indexing with new course materials.
  - Added `3b_Dynamics.pdf`.
  - Added `4a_Convolution.pdf`.
  - Incremental indexing processed only the two new PDFs.
- Added Streamlit demo UI in `streamlit_app.py`.
  - Question input.
  - Top-k control.
  - Answer display.
  - Retrieval query display.
  - Citations table.
  - Expandable retrieved contexts.
  - Optional query logging.
- Added shared RAG pipeline in `src/pipeline.py` so CLI and Streamlit use the same business logic.

### Changed

- Reworked `README.md` into a GitHub / resume showcase format.
- Added Chinese explanatory comments to core source modules.
- Added `streamlit==1.52.2` to `requirements.txt`.

### Verified

- Incremental indexing:
  ```text
  Indexed files: 2
  skipped files: 7
  chunks added: 47
  ```
- Updated index status:
  ```text
  indexed_files: 9
  collection_count: 216
  ```
- New lecture retrieval:
  ```text
  What are convolutional networks?
  ```
  retrieved `4a_Convolution.pdf`.
- Test result:
  ```text
  13 passed
  ```

## 2026-06-16

### Added

- Implemented multi-query retrieval.
  - Non-English questions can be expanded into multiple English retrieval queries.
  - Each retrieval query is searched separately.
  - Retrieved chunks are merged, deduplicated, and ranked.
  - Query logs now record `retrieval_queries` and per-chunk `matched_queries`.
- Added query rewrite tests and multi-query retrieval tests.
- Added lightweight retrieval regression evaluation.
  - Evaluation cases are stored in `data/eval/retrieval_cases.yaml`.
  - New CLI command: `python -m src.app eval-retrieval --top-k 6`.
  - Reports source hit and page hit for each case.

### Changed

- Retrieval no longer depends on a single rewritten query.
- README now documents multi-query retrieval and retrieval language behavior.
- Grounded prompt now reminds users to upload or add relevant course material when retrieved context is insufficient.

### Verified

- Query:
  ```text
  请问什么是交叉熵，用英文回答
  ```
- Retrieval queries:
  ```text
  cross entropy loss
  cross entropy
  cross entropy derivation
  cross entropy KL divergence
  ```
- Correctly retrieved cross-entropy pages from `3a_Variations.pdf`.
- Test result:
  ```text
  13 passed
  ```
- Retrieval regression result:
  ```text
  cases: 5
  source_recall@6: 1.00
  page_recall@6: 1.00
  ```

### Notes

- DeepSeek API key was rotated after a previous key became invalid.
- Full query generation depends on DeepSeek; glossary-based expansion works before calling the LLM planner.

## 2026-06-15

### Added

- Created project virtual environment at `.venv`.
- Installed phase 1 dependencies.
- Added `requirements.txt`, `.gitignore`, `.env.example`, and local `.env` template.
- Installed and verified Ollama with `nomic-embed-text`.
- Added phase 1 RAG implementation:
  - PDF ingestion with PyMuPDF.
  - Fixed-size chunking.
  - Ollama embedding provider.
  - DeepSeek OpenAI-compatible chat provider.
  - Chroma vector storage with explicit embeddings.
  - Incremental indexing using file hashes.
  - CLI commands: `index`, `status`, `ask`.
  - Query logging to `logs/queries/YYYY-MM-DD.jsonl`.
- Added initial query rewrite for Chinese-to-English retrieval.
- Added language-following behavior:
  - Answer in the user's language by default.
  - If the user explicitly requests a response language, use that language.

### Verified

- Indexed 7 COMP9444 PDFs.
- Collection count:
  ```text
  169 chunks
  ```
- Incremental indexing:
  ```text
  Indexed files: 0; skipped files: 7; chunks added: 0
  ```
- Sample query:
  ```text
  How do you implement a neural network in PyTorch?
  ```
  correctly retrieved `2c_PyTorch.pdf`.
- Query log writes JSONL records and does not store API keys.

### Notes

- Chroma's default ONNX embedding function writes to `~/.cache/chroma/...`, which may be blocked in the managed workspace.
- The project uses explicit embeddings instead of Chroma's default embedding function.
- Local Ollama HTTP calls may require elevated sandbox permissions in this environment.
