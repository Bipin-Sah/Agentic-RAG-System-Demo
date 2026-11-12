"""Retrieval agent: decides whether to search, what to search for, and answers with citations.

CLI: ``python -m rag.agent "your question"``
"""

from __future__ import annotations

import argparse
import sys
import time

from rag.answer import Answer, SearchStep
from rag.citations import (
    CITATION_RULES,
    SourceRegistry,
    format_sources,
    resolve_citations,
    strip_all_labels,
    strip_unknown_labels,
)
from rag.llm import ChatClient, ChatResult, assistant_message
from rag.retriever import Retriever

SEARCH_TOOL_NAME = "search_knowledge_base"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Semantic search over the user's uploaded documents. Returns the most relevant "
            "passages, each labelled with a source id like S1."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused search query: key entities and terms.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of passages to return (1-10).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = f"""You are a precise research assistant answering questions about the user's \
documents. You can search them with the `{SEARCH_TOOL_NAME}` tool.

Deciding whether to retrieve:
- Do NOT search for greetings, small talk, questions about yourself, or requests that need no \
document knowledge (e.g. arithmetic, rephrasing your previous answer). Answer those directly.
- Search for anything that may be covered by the documents (facts, names, numbers, policies, \
procedures, specifications), even if you think you know the answer.

How to search:
- Write focused keyword-style queries rather than repeating the user's question verbatim.
- Break multi-part or comparative questions into one search per part.
- If results are missing or off-topic, reformulate (synonyms, more specific terms) and search \
again. You may search at most {{max_searches}} times.

Answering:
- Base document-related answers only on retrieved passages. {CITATION_RULES}
- Be concise and directly answer the question."""


def _scrub_history(history: list[dict] | None) -> list[dict]:
    """Drop [S#] labels from earlier turns: labels are per-answer and would be misread."""
    return [{**m, "content": strip_all_labels(m.get("content") or "")} for m in history or []]


class RetrievalAgent:
    def __init__(
        self,
        chat: ChatClient,
        retriever: Retriever,
        max_searches: int = 4,
        default_k: int = 5,
    ) -> None:
        self.chat = chat
        self.retriever = retriever
        self.max_searches = max_searches
        self.default_k = default_k

    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        """Answer ``question``. ``history`` is prior turns as [{"role", "content"}, ...]."""
        start = time.perf_counter()
        result = Answer(question=question, answer="")
        registry = SourceRegistry()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(max_searches=self.max_searches)},
            *_scrub_history(history),
            {"role": "user", "content": question},
        ]

        final: ChatResult | None = None
        while True:
            budget_left = len(result.searches) < self.max_searches
            response = self._call_llm(
                result, messages, tool_choice="auto" if budget_left else "none"
            )
            if not response.tool_calls:
                final = response
                break
            messages.append(assistant_message(response))
            for call in response.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": self._run_tool(call.name, call.arguments, result, registry),
                    }
                )

        result.answer = strip_unknown_labels(final.content or "", registry)
        result.retrieved = registry.chunks
        result.citations = resolve_citations(result.answer, registry)
        result.total_ms = (time.perf_counter() - start) * 1000
        return result

    def _call_llm(self, result: Answer, messages: list[dict], tool_choice: str) -> ChatResult:
        t0 = time.perf_counter()
        response = self.chat.complete(messages, tools=[SEARCH_TOOL], tool_choice=tool_choice)
        result.llm_ms += (time.perf_counter() - t0) * 1000
        result.llm_calls += 1
        result.prompt_tokens += response.prompt_tokens
        result.completion_tokens += response.completion_tokens
        if tool_choice == "none" and response.tool_calls:
            # The model ignored tool_choice="none"; treat its text (if any) as final.
            response = ChatResult(content=response.content or "", tool_calls=[])
        return response

    def _run_tool(
        self, name: str, arguments: dict, result: Answer, registry: SourceRegistry
    ) -> str:
        if name != SEARCH_TOOL_NAME:
            return f"Error: unknown tool {name!r}."
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        if len(result.searches) >= self.max_searches:
            return "Search budget exhausted. Answer with the passages you already have."
        try:
            k = max(1, min(10, int(arguments.get("k") or self.default_k)))
        except (TypeError, ValueError):
            k = self.default_k
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(query, k)
        ms = (time.perf_counter() - t0) * 1000
        result.retrieval_ms += ms
        result.searches.append(SearchStep(query=query, num_results=len(chunks), ms=ms))
        return format_sources(registry.add(chunks))


def print_answer(ans: Answer) -> None:
    print(ans.answer)
    if ans.citations:
        print("\nCitations:")
        for c in ans.citations:
            print(f"  [{c.label}] {c.chunk.location}")
    if ans.searches:
        print("\nSearches:")
        for s in ans.searches:
            print(f"  - {s.query!r} -> {s.num_results} results ({s.ms:.0f} ms)")
    print(
        f"\nLatency: total {ans.total_ms:.0f} ms | retrieval {ans.retrieval_ms:.0f} ms | "
        f"LLM {ans.llm_ms:.0f} ms over {ans.llm_calls} call(s)"
    )


def main(argv: list[str] | None = None) -> int:
    from rag.config import Settings
    from rag.factory import build_agent

    parser = argparse.ArgumentParser(description="Ask the retrieval agent a question.")
    parser.add_argument("question")
    args = parser.parse_args(argv)
    print_answer(build_agent(Settings.from_env()).answer(args.question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
