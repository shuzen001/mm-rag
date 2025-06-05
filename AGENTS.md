# AGENTD: Contribution Guidelines and Setup

This file provides a quick overview for navigating the project, running tests, and adhering to basic development standards.

## Project Navigation
- `app.py` – FastAPI entry point for handling uploads and building the FAISS vector store.
- `main.py` – Standalone script that runs the question-answer workflow.
- `utils/` – Helper modules:
  - `LLM_Tool.py` – Wrapper around GPT models and embeddings.
  - `extract_file_utils.py` – Document extraction utilities.
  - `summarize.py` – Summary and image-handling logic.
  - `vector_store.py` – Image management and FAISS retriever creation.
- `tests/` – Pytest-based tests (currently only `tests/test_utils.py`).

## Recommended Commands
**Setup**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
**Running the API**
```bash
uvicorn app:app --host 0.0.0.0 --port 1230 --reload
```
**Executing the main workflow**
```bash
python main.py
```
**Running Tests**
```bash
pytest -q
```

## Development Standards
- Python version: 3.10+
- Use `isort` and `black` for formatting.
- Write new tests with `pytest` and place them under `tests/`.
- When adding dependencies, update `requirements.txt`.
- Keep environment variables (like `OPENAI_API_KEY`) in an `.env` file not committed to the repository.
