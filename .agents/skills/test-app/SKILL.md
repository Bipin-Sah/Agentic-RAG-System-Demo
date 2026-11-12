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

## 5. Ingestion smoke test (needs OPENAI_API_KEY + DB)

```bash
echo "The Zephyr X2 drone has a 47 minute battery life." > /tmp/smoke.txt
python -m rag.ingest /tmp/smoke.txt          # prints chunks + doc_id
python -m rag.ingest /tmp/smoke.txt          # second run must say "skipped (already ingested)"
python -m rag.ingest --reset path/to/dir     # wipe store, ingest every pdf/docx/txt/md in dir
```

Without `OPENAI_API_KEY` the CLI exits with a clear `MissingAPIKey` error (expected).

## 6. Retrieval agent smoke test (needs OPENAI_API_KEY + ingested docs)

```bash
python -m rag.agent "hi, who are you?"                       # expect: no "Searches:" section
python -m rag.agent "How long does the Zephyr X2 battery last?"  # expect: >=1 search + [S#] citations
```

Check that the output shows the citations list (`[S1] source, chunk N`), the search queries the
agent chose, and the latency breakdown.

## 7. Eval harness (needs OPENAI_API_KEY + DB)

The fixed question set is `evals/questions.jsonl` (single-hop, multi-hop, no-retrieval and
unanswerable questions) over the fictional corpus in `data/sample_corpus/`. The eval uses its own
pgvector schema (`eval`), so it never mixes with documents you ingested yourself.

```bash
python -m evals.run_eval --ingest                  # first run: ingest corpus (deduped), then evaluate
python -m evals.run_eval --per-question            # also print per-question results
python -m evals.run_eval --limit 3 --systems agent # quick smoke run
python -m evals.run_eval --output eval_results/run.json   # full answers as JSON (git-ignored)
```

Expect live progress lines, then a comparison table (agent vs baseline) with answer accuracy,
cited-source recall, retrieve/skip decision accuracy, refusal rate on unanswerable questions,
searches / LLM calls / tokens per question, and mean / p95 latency. `Errors` should be 0.
Without `OPENAI_API_KEY` it exits with code 2 and a clear message. The offline version of this
pipeline (fake LLM + embedder) runs in `tests/test_eval.py`.

To regenerate the PDF/DOCX files in the corpus: `python scripts/make_sample_corpus.py`.

## 8. UI end-to-end test (needs OPENAI_API_KEY + DB) — required before any PR touching `app/`

```bash
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
curl -s localhost:8501/_stcore/health     # -> ok
```

Open http://localhost:8501 in the browser and record the following steps:

1. Page shows the title "Agentic RAG" and the sidebar shows "N documents · M chunks". Without
   `OPENAI_API_KEY` you get a red "Setup problem" box instead (that's expected).
2. **Upload**: in the sidebar, upload `data/sample_corpus/zephyr_x2_spec.docx` (or any PDF/DOCX/TXT),
   then click **Ingest**. Expect a green "`<file>`: N chunks in X.Xs" message and the file in the
   document list. Clicking Ingest again on the same file should say "already ingested".
3. **Agent, no retrieval**: with mode "Agentic RAG", ask `Hi! What can you do?`. Expect an answer,
   a grey latency line (`Agentic RAG · total …s · retrieval 0.00s · … · 0 searches`) and
   "Answered without retrieval."
4. **Agent with retrieval**: ask `What is the maximum payload of the Zephyr X2?`. Expect 2.3 kg
   with an `[S1]` style citation, a **Sources** expander showing `zephyr_x2_spec.docx` and a
   snippet, and a **Retrieval trace** expander listing the search queries.
5. **Follow-up**: ask `And how long can it fly?`. The answer should resolve "it" to the X2.
6. **Baseline**: switch the mode to "Naive RAG (baseline)" and ask the payload question again.
   The latency line should show exactly 1 search and 1 LLM call.
7. **Latency tab**: shows Last / Mean / p50 / p95 metrics, a retrieval vs LLM bar chart, and one
   table row per question with mode, searches, LLM calls and tokens.
8. **Delete**: the ✕ next to a document removes it from the list. **Clear chat** empties the chat
   and the latency table.

Offline UI checks (fake components, no key needed) run in `tests/test_streamlit_app.py`.
