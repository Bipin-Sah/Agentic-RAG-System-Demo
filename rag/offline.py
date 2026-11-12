"""Deterministic, network-free stand-ins for the OpenAI embedder and chat model.

Used by the test suite and by the opt-in offline demo mode (``RAG_OFFLINE=1``), which lets the
whole app run without an API key. Answers are extractive and far weaker than a real LLM's.
"""

from __future__ import annotations

import hashlib
import math
import re

from rag.llm import ChatResult, ToolCall

_TOKEN = re.compile(r"[a-z0-9]+")
_SOURCE = re.compile(r"\[(S\d+)\] \([^\n]*\)\n(.*?)(?=\n\n\[S\d+\] \(|\Z)", re.S)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_GREETING = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b", re.I
)
_PRONOUNS = {"it", "its", "they", "them", "their", "that", "this", "those", "these", "he", "she"}
_STOPWORDS = _PRONOUNS | {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "me", "much", "many", "of", "on", "or", "our", "the", "to", "was", "we",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your", "long",
}  # fmt: skip


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_words(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in _STOPWORDS}


class HashingEmbedder:
    """Deterministic bag-of-words embedder: shared words => higher cosine similarity."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokens(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        if not any(vec):
            vec[0] = 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class RuleChatClient:
    """Rule-based chat model speaking the same protocol as OpenAIChatClient.

    - greetings are answered directly (no retrieval);
    - otherwise it calls the search tool, folding the previous question into follow-ups that
      use pronouns ("how long can it fly?");
    - given sources, it quotes the sentences that best overlap the question, with citations.
    """

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> ChatResult:
        last = messages[-1]
        content = last.get("content") or ""
        if last["role"] == "tool":
            return self._answer(self._current_question(messages), content)
        if content.startswith("Sources:\n"):
            sources, _, question = content[len("Sources:\n") :].rpartition("\n\nQuestion: ")
            return self._answer(question, sources)
        if _GREETING.match(content) or not tools:
            return ChatResult(
                content="Hello! Upload documents and ask me about them; I'll cite my sources.",
                prompt_tokens=20,
                completion_tokens=15,
            )
        query = self._search_query(messages, content)
        call = ToolCall("call_1", tools[0]["function"]["name"], {"query": query})
        return ChatResult(content=None, tool_calls=[call], prompt_tokens=50, completion_tokens=10)

    @staticmethod
    def _current_question(messages: list[dict]) -> str:
        return next(m["content"] for m in reversed(messages) if m["role"] == "user")

    @staticmethod
    def _search_query(messages: list[dict], question: str) -> str:
        if not _PRONOUNS & set(tokens(question)):
            return question
        earlier = [m["content"] for m in messages[:-1] if m["role"] == "user"]
        return f"{earlier[-1]} {question}" if earlier else question

    @staticmethod
    def _answer(question: str, sources: str) -> ChatResult:
        wanted = content_words(question)
        scored = []
        for label, body in _SOURCE.findall(sources):
            for sentence in _SENTENCE.split(body):
                sentence = sentence.strip(" -*#")
                overlap = len(wanted & content_words(sentence))
                if overlap and len(sentence) > 3:
                    scored.append((overlap, -len(scored), sentence, label))
        if not scored:
            return ChatResult(content="I couldn't find that in the documents.", prompt_tokens=100)
        best = sorted(scored, reverse=True)[:2]
        text = " ".join(f"{s.rstrip('.')}. [{label}]" for _, _, s, label in best)
        return ChatResult(content=text, prompt_tokens=len(sources) // 4, completion_tokens=40)
