# COMP9444 RAG Assistant

A retrieval-augmented generation assistant for UNSW COMP9444 course materials. It answers questions over local course PDFs with citations, multi-query retrieval, bilingual query rewriting, query logs, and lightweight retrieval regression tests.

The project is designed as an interview-ready RAG implementation: small enough to explain end to end, but complete enough to demonstrate real engineering tradeoffs.

See [CHANGELOG.md](CHANGELOG.md) for iteration history and verification notes.

## Documentation

- [User Guide](docs/USER_GUIDE.md): how to set up, index materials, ask questions, and use the Streamlit demo.
- [Development Process](docs/DEVELOPMENT_PROCESS.md): how the project was built from scratch and why each feature was added.
- [Code Reading Guide](docs/CODE_READING_GUIDE.md): recommended order for reading the codebase.
- [Resume Project](docs/RESUME_PROJECT.md): Chinese resume-ready project description.

## Highlights

- PDF ingestion with page-level metadata using PyMuPDF.
- Local vector search with Chroma and explicit embeddings.
- Low-cost model setup: Ollama `nomic-embed-text` for embeddings, DeepSeek API for answer generation.
- Provider abstraction for API/local chat and embedding models.
- Multi-query retrieval: one user question can produce several English retrieval phrases, searched and merged before generation.
- Bilingual retrieval: Chinese questions are rewritten into English retrieval queries because the slides are in English.
- Citation-first answers with source file, page, and chunk metadata.
- Query logs with retrieval queries, matched chunks, distances, and answer traces.
- Incremental indexing with file hashes.
- Lightweight retrieval regression evaluation.
- CLI and Streamlit demo.

## Architecture

```text
COMP9444 PDFs
→ PyMuPDF page extraction
→ fixed-size chunking
→ Ollama embeddings
→ Chroma vector store
→ query rewrite / multi-query retrieval
→ context merge + dedup
→ DeepSeek grounded answer
→ citations + query log
```

Core modules:

```text
src/
├── app.py              # Typer CLI
├── chunk.py            # text normalization and chunking
├── config.py           # config, manifest, env loading
├── evaluate.py         # lightweight retrieval evaluation
├── generate.py         # grounded prompt and answer generation
├── index.py            # Chroma indexing and incremental state
├── ingest.py           # PDF parsing
├── logging_utils.py    # JSONL query logs
├── pipeline.py         # shared RAG pipeline
├── providers.py        # chat / embedding provider abstraction
├── query_rewrite.py    # bilingual and multi-query planning
└── retrieve.py         # single and multi-query retrieval
```

## Current Verification

Local COMP9444 test set:

```text
1a_Overview.pdf
1b_Neuroanatomy.pdf
1c_Perceptrons.pdf
2a_Probability.pdf
2b_Generalization.pdf
2c_PyTorch.pdf
3a_Variations.pdf
3b_Dynamics.pdf
4a_Convolution.pdf
```

Verified locally:

```text
Indexed files: 9
Chunks added: 216
Collection count: 216
Unit tests: 13 passed
Retrieval evaluation: source_recall@6 = 1.00, page_recall@6 = 1.00
```

Example query:

```text
请问什么是交叉熵，用英文回答
```

Generated retrieval queries:

```text
cross entropy loss
cross entropy
cross entropy derivation
cross entropy KL divergence
```

The system retrieves cross-entropy pages from `3a_Variations.pdf` and answers in English as requested.

## Setup

Create and activate the project virtual environment:

```bash
cd rag_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Tested with:

```text
Python 3.13.9
```

## Model Configuration

Create a local `.env` from `.env.example`:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-v4-flash

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Install and start Ollama, then pull the embedding model:

```bash
ollama pull nomic-embed-text
ollama list
```

Default phase-1 setup:

```text
Chat model: DeepSeek API, deepseek-v4-flash
Embedding model: Ollama, nomic-embed-text
```

The provider layer is config-driven, so the project can also support API embeddings or fully local Ollama chat models.

## Data

Put COMP9444 PDFs here:

```text
data/raw/comp9444/
```

Course materials are ignored by git. The repository should include code, configs, docs, and small evaluation metadata, not copyrighted PDFs or vector indexes.

## Index And Query

Index new or changed PDFs:

```bash
python -m src.app index
```

Check index status:

```bash
python -m src.app status
```

Ask a question:

```bash
python -m src.app ask "How do you implement a neural network in PyTorch?" --top-k 4
```

Run retrieval regression checks:

```bash
python -m src.app eval-retrieval --top-k 6
```

The second index run should skip unchanged PDFs:

```text
Indexed files: 0; skipped files: 9; chunks added: 0
```

## Streamlit Demo

Launch the web demo:

```bash
streamlit run streamlit_app.py
```

The demo includes:

- question input;
- top-k control;
- answer display;
- generated retrieval queries;
- citations table;
- expandable retrieved contexts;
- optional query logging.

## Language And Retrieval Behavior

- If the user asks in English, the assistant answers in English.
- If the user asks in Chinese, the assistant answers in Chinese.
- If the user explicitly requests a response language, that requested language takes priority.
- Since COMP9444 slides are in English, non-English questions are rewritten into English retrieval queries before embedding search.
- The final answer still uses the original user question, so language preference and intent are preserved.
- If retrieved course context is insufficient, the assistant states that the available material is insufficient and reminds the user that they may need to upload or add relevant course material.

## Query Logs

Each query is logged by default to:

```text
logs/queries/YYYY-MM-DD.jsonl
```

Logs include:

- original question;
- retrieval queries;
- matched queries per chunk;
- answer;
- citations;
- retrieval distances;
- short context previews;
- provider names.

Logs do not store API keys and are ignored by git.

Disable logging for a one-off CLI query:

```bash
python -m src.app ask "What is cross entropy?" --no-log
```

## Chroma Embedding Cache Note

Chroma's default ONNX embedding function tries to download/cache its model under:

```text
~/.cache/chroma/...
```

This project avoids that path by supplying embeddings explicitly through the configured embedding provider.

## Example Resume Description

```text
Built a RAG-based course assistant for COMP9444 materials using PyMuPDF, Chroma, Ollama embeddings, and DeepSeek API; implemented multi-query retrieval, bilingual query rewriting, citation logging, incremental indexing, and retrieval regression evaluation, achieving 100% source/page recall@6 on a 5-case validation set.
```
