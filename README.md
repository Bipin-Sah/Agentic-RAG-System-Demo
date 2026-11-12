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

## Evaluate agent vs. naive RAG

```bash
python -m evals.run_eval --ingest --per-question
```

Runs the fixed question set in `evals/questions.jsonl` against the retrieval agent and a naive
single-shot RAG baseline (always retrieve top-k for the raw question, one LLM call) and prints a
comparison table: answer accuracy, cited-source recall, retrieve/skip decisions, refusals on
unanswerable questions, cost (searches, LLM calls, tokens) and latency.
