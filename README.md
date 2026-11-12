# Agentic RAG System Demo

An end-to-end agentic Retrieval-Augmented Generation (RAG) system: document
ingestion into a local pgvector store, a retrieval agent that decides when and
what to retrieve and cites its sources, an evaluation harness against a naive
RAG baseline, and a Streamlit chat UI.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set OPENAI_API_KEY (or export it)
docker compose up -d db         # local Postgres + pgvector on :5433
pytest -q
```

See [AGENTS.md](AGENTS.md) for project structure and conventions, and
[.agents/skills/test-app/SKILL.md](.agents/skills/test-app/SKILL.md) for the full
install / run / test procedure.
