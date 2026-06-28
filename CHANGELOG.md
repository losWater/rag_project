# Changelog

This file records project iterations so regressions can be traced back to the likely change that introduced them.

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
