# AGENTS.md

Guidance for humans and AI agents working in this repository.

## What this project is

An agentic RAG system:

1. **Ingestion** – load PDF / DOCX / text files, chunk, embed with OpenAI, store in Postgres + pgvector.
2. **Retrieval agent** – an OpenAI tool-calling agent that decides *whether* to retrieve, *what* to
   search for (possibly several times), and answers with citations to source chunks.
3. **Eval harness** – runs a fixed question set against the agent and a naive single-shot RAG
   baseline and prints a comparison table.
4. **Web UI** – Streamlit chat app: upload documents, ask questions, see citations and latency.

## Project structure

```
rag/                    # Python package with all core logic
  config.py             # Settings loaded from env / .env
  ...                   # loaders, chunking, embeddings, store, agent, baseline (added per slice)
app/                    # Streamlit UI
evals/                  # Eval harness + fixed question set
data/sample_corpus/     # Small fictional corpus used by the eval and demos
tests/                  # pytest suite (offline by default)
.agents/skills/test-app # How to install, run and test the app (read before every PR)
docker-compose.yml      # Local pgvector database (host port 5433)
requirements.txt        # All runtime + dev dependencies
```

## Conventions

- **Python 3.10+**, type hints on public functions, small focused modules.
- **Tests are required.** Every change to core logic (`rag/`, `evals/`) ships with pytest tests.
  - Unit tests must run **offline**: use the fake embedder / fake chat client in `tests/`, never the
    real OpenAI API.
  - Tests that need Postgres are marked `@pytest.mark.db` and are skipped automatically when the
    database is unreachable.
- **No secrets in code.** API keys and credentials come from environment variables (or a local,
  git-ignored `.env`; see `.env.example`). Never commit `.env`, keys, tokens or real customer data.
- **Lint before PRs.** `ruff check .` and `ruff format --check .` must pass, as must `pytest`.
- **Small commits, one PR per working slice.** Each PR should leave `main` in a working state.
- **UI changes:** before opening a PR that touches `app/`, run the app and test it end-to-end
  following `.agents/skills/test-app/SKILL.md`, and attach a screen recording to the PR.
- Configuration lives in `rag/config.py`; do not read `os.environ` ad hoc elsewhere.

## Common commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d db          # start pgvector on localhost:5433
ruff check . && ruff format --check .
pytest -q
```

See `.agents/skills/test-app/SKILL.md` for the full install / run / test procedure.
