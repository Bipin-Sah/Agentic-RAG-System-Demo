---
name: test-app
description: How to install, run, lint and test the Agentic RAG app (pgvector DB, pytest, Streamlit UI). Use before opening any PR in this repo.
---

# Testing the Agentic RAG app

Run this checklist before **every** PR. For PRs that touch the UI (`app/`), also do the
"UI end-to-end test" section and record it.

## 1. Install

```bash
cd <repo root>
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp -n .env.example .env   # then set OPENAI_API_KEY in .env, or export it in the shell
```

`OPENAI_API_KEY` must be set (env var or `.env`) for anything that calls OpenAI
(real ingestion, the agent, the eval harness, the UI). The unit tests do **not** need it.

## 2. Start the vector store

```bash
docker compose up -d db
# wait until healthy
until docker exec agentic-rag-db pg_isready -U rag -d rag >/dev/null 2>&1; do sleep 1; done
```

- Postgres + pgvector listens on `localhost:5433` (user/password/db: `rag`/`rag`/`rag`).
- Default `DATABASE_URL=postgresql://rag:rag@localhost:5433/rag`.
- Reset all data: `docker compose down -v && docker compose up -d db`.

## 3. Lint

```bash
ruff check .
ruff format --check .      # use `ruff format .` to fix
```

## 4. Unit tests

```bash
pytest -q
```

- Tests run offline with fake embedder/LLM; no OpenAI key needed.
- Tests marked `db` use the pgvector database and are **skipped** if it is unreachable, so make
  sure step 2 is done and confirm in the pytest summary that nothing was skipped unexpectedly
  (`pytest -q -rs` shows skip reasons).
