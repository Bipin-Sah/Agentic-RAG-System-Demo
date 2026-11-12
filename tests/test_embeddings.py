from types import SimpleNamespace

from rag.embeddings import OpenAIEmbedder


class FakeEmbeddingsAPI:
    def __init__(self):
        self.requests = []

    def create(self, model, input, dimensions):
        self.requests.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), float(dimensions)])
            for i, t in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))


def test_openai_embedder_batches_and_orders():
    api = FakeEmbeddingsAPI()
    emb = OpenAIEmbedder(model="m", dim=2, batch_size=2, client=SimpleNamespace(embeddings=api))
    vectors = emb.embed(["a", "bb", "ccc", "", "eeeee"])
    assert [len(r) for r in api.requests] == [2, 2, 1]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 1.0, 5.0]  # "" sent as " "
