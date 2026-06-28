# COMP9444 RAG Project Requirements

## Current Environment Check

Checked on: 2026-06-15

Current Python:

```text
Python 3.13.9
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
pip 25.2
```

Virtual environment status:

```text
venv = False
```

This means the current Python is the system/global Python, not a project-specific virtual environment.

## Created Project Virtual Environment

Created:

```text
rag_project/.venv
```

Verified:

```text
Python 3.13.9
venv = True
```

Phase 1 dependencies installed successfully in `.venv`.

## Installed / Missing Packages

Important packages currently missing:

```text
pypdf: no
pymupdf / fitz: no
chromadb: no
langchain: no
langchain-chroma: no
langchain-openai: no
python-dotenv: no
click: no
rich: no
```

After creating `.venv`, the required phase 1 packages were installed successfully:

```text
pymupdf==1.27.2.3
chromadb==1.5.9
langchain==1.3.9
langchain-chroma==1.1.0
langchain-openai==1.3.2
python-dotenv==1.2.2
PyYAML==6.0.3
typer==0.25.1
rich==15.0.0
pytest==9.1.0
```

Already available:

```text
PyYAML: yes
```

There are also global packages installed, including Jupyter, NumPy, PyTorch, TorchVision, and TorchAudio. These should not be used as the project dependency baseline because they are installed globally.

## Recommendation

Create a virtual environment for this project.

Reason:

- Avoid polluting the global Python installation.
- Keep this RAG project independent from future CV / LLM fine-tuning projects.
- Make dependency versions reproducible for README and interview discussion.
- Reduce the risk that global Jupyter / PyTorch packages affect this project unexpectedly.

Recommended location:

```text
rag_project/.venv
```

Recommended commands:

```bash
cd rag_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Note: the current Python version is 3.13.9. The phase 1 dependency set installed successfully, so there is no immediate need to switch to Python 3.12.

## Smoke Test Results

Passed:

- All key imports worked: PyMuPDF, Chroma, LangChain, LangChain-Chroma, LangChain-OpenAI, dotenv, PyYAML, Typer, Rich, Pytest.
- PyMuPDF successfully opened `data/raw/comp9444/1a_Overview.pdf`.
- PyMuPDF extracted text from page 1.
- Chroma local persistence worked when embeddings were supplied explicitly.

Important caveat:

- Chroma's default ONNX embedding function currently tries to write to `~/.cache/chroma/...`.
- In this sandbox, writing to `~/.cache` is blocked.
- For phase 1, avoid relying on Chroma's default embedding function. Use an explicit embedding provider such as OpenAI embeddings, or configure/cache embeddings inside the project directory.

## Phase 1 Dependencies

Minimal dependencies for the first RAG milestone:

```text
pymupdf
chromadb
langchain
langchain-chroma
langchain-openai
python-dotenv
pyyaml
typer
rich
pytest
```

Why these packages:

- `pymupdf`: extract text from COMP9444 PDF slides page by page.
- `chromadb`: local vector database for document chunks.
- `langchain`: RAG orchestration primitives.
- `langchain-chroma`: LangChain integration for Chroma.
- `langchain-openai`: OpenAI-compatible embedding / chat model integration.
- `python-dotenv`: load API keys and config from `.env`.
- `pyyaml`: read `configs/rag.yaml`, `manifest.yaml`, and `retrieval_rules.yaml`.
- `typer`: build a small CLI for indexing and asking questions.
- `rich`: readable CLI output and debugging.
- `pytest`: tests for parsing, chunking, manifest loading, and incremental indexing.

## Optional Later Dependencies

These are not needed for phase 1:

```text
ragas
streamlit
gradio
sentence-transformers
rerankers
```

Use later when adding evaluation, web demo, local embeddings, or reranking.
