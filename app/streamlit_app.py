"""Streamlit chat UI: upload documents, ask questions, inspect citations and latency.

streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run` puts app/ on sys.path, not the repo root
    sys.path.insert(0, str(ROOT))

from app.services import (  # noqa: E402
    UPLOAD_TYPES,
    Components,
    citation_rows,
    history_for_model,
    ingest_upload,
    latency_caption,
    latency_record,
    latency_rows,
    latency_summary,
)

UPLOAD_DIR = ROOT / "uploads"
AGENT, BASELINE = "Agentic RAG", "Naive RAG (baseline)"

CSS = """
<style>
.block-container {max-width: 980px; padding-top: 2rem;}
.latency {color: #6b7280; font-size: 0.82rem; margin-top: -0.4rem;}
.cite {border-left: 3px solid #6366f1; padding: 0.25rem 0.75rem; margin: 0.4rem 0;
       background: rgba(99,102,241,0.06); border-radius: 4px; font-size: 0.88rem;}
.cite b {color: #4f46e5;}
</style>
"""


@st.cache_resource(show_spinner="Connecting to the vector store…")
def _build_components() -> Components:
    from rag.config import Settings
    from rag.factory import build_agent, build_baseline, build_embedder, build_store
    from rag.ingest import Ingestor

    settings = Settings.from_env()
    embedder = build_embedder(settings)  # raises MissingAPIKey before touching the DB
    store = build_store(settings)
    return Components(
        store=store,
        ingestor=Ingestor(store, embedder, settings.chunk_size, settings.chunk_overlap),
        systems={AGENT: build_agent(settings, store), BASELINE: build_baseline(settings, store)},
        offline=settings.offline,
    )


def get_components() -> Components | None:
    if "components" in st.session_state:  # injected by tests
        return st.session_state.components
    try:
        return _build_components()
    except Exception as exc:  # missing key, DB down, ...
        st.error(f"**Setup problem:** {exc}")
        st.info(
            "Set `OPENAI_API_KEY` (env or `.env`) and start the DB with "
            "`docker compose up -d db`, then reload. See `.agents/skills/test-app/SKILL.md`."
        )
        return None


def render_answer(msg: dict) -> None:
    st.markdown(msg["content"])
    answer = msg.get("answer")
    if answer is None:
        return
    st.markdown(
        f"<div class='latency'>{msg['mode']} · {latency_caption(answer)}</div>",
        unsafe_allow_html=True,
    )
    rows = citation_rows(answer)
    if rows:
        with st.expander(f"Sources ({len(rows)})"):
            for r in rows:
                st.markdown(
                    f"<div class='cite'><b>[{r['label']}]</b> {r['location']}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(r["snippet"])
    if answer.searches:
        with st.expander(f"Retrieval trace ({len(answer.searches)} searches)"):
            for s in answer.searches:
                st.markdown(f"- `{s.query}` → {s.num_results} results · {s.ms:.0f} ms")
    elif msg["mode"] == AGENT:
        st.caption("Answered without retrieval.")


def sidebar(components: Components) -> str:
    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader(
            "Upload PDF, DOCX or text files", type=UPLOAD_TYPES, accept_multiple_files=True
        )
        if st.button("Ingest", type="primary", disabled=not files, use_container_width=True):
            for f in files:
                with st.spinner(f"Ingesting {f.name}…"):
                    try:
                        r = ingest_upload(components.ingestor, f.name, f.getvalue(), UPLOAD_DIR)
                    except Exception as exc:
                        st.error(f"{f.name}: {exc}")
                        continue
                if r.skipped:
                    st.info(f"{r.source}: already ingested")
                else:
                    st.success(f"{r.source}: {r.num_chunks} chunks in {r.seconds:.1f}s")

        docs = components.store.list_documents()
        st.caption(f"{len(docs)} documents · {components.store.count_chunks()} chunks")
        for d in docs:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"<small>{d.source} ({d.num_chunks})</small>", unsafe_allow_html=True)
            if col2.button("✕", key=f"del-{d.id}", help=f"Delete {d.source}"):
                components.store.delete_document(d.id)
                st.rerun()

        st.divider()
        mode = st.radio("Answering mode", list(components.systems), key="mode")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.latency = []
            st.rerun()
    return mode


def chat_tab(components: Components, mode: str) -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg)
            else:
                st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return
    history = history_for_model(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = components.systems[mode].answer(question, history=history)
            except Exception as exc:
                st.error(f"Failed to answer: {exc}")
                return
        msg = {"role": "assistant", "content": answer.answer, "answer": answer, "mode": mode}
        st.session_state.messages.append(msg)
        turn = len(st.session_state.latency) + 1
        st.session_state.latency.append(latency_record(turn, mode, answer))
        render_answer(msg)


def latency_tab() -> None:
    records = st.session_state.latency
    if not records:
        st.info("Ask a question to start tracking latency.")
        return
    summary = latency_summary(records)
    cols = st.columns(4)
    cols[0].metric("Last", f"{summary['last_ms']:,.0f} ms")
    cols[1].metric("Mean", f"{summary['mean_ms']:,.0f} ms")
    cols[2].metric("p50", f"{summary['p50_ms']:,.0f} ms")
    cols[3].metric("p95", f"{summary['p95_ms']:,.0f} ms")
    rows = latency_rows(records)
    st.bar_chart(
        {
            "retrieval_ms": [r["retrieval_ms"] for r in rows],
            "llm_ms": [r["llm_ms"] for r in rows],
        },
        x_label="turn",
        y_label="ms",
    )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Agentic RAG", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Agentic RAG")
    st.caption("Upload documents, ask questions, and compare the agent with a naive baseline.")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("latency", [])

    components = get_components()
    if components is None:
        return
    if components.offline:
        st.warning(
            "**Offline demo mode** (`RAG_OFFLINE=1`): hashing embeddings and a rule-based "
            "extractive model stand in for OpenAI, so answers are crude quotes, not LLM output."
        )
    mode = sidebar(components)
    chat, latency = st.tabs(["Chat", "Latency"])
    with chat:
        chat_tab(components, mode)
    with latency:
        latency_tab()


main()
