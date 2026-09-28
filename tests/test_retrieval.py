import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from retrieval import Chunk, METRICS, answer_question, build_context, chunk_documents, retrieve, vector_scores


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [Chunk("A", "first evidence", {"source": "test"}), Chunk("B", "second evidence", {})]

    def test_known_scores_and_different_rankings(self):
        scores = vector_scores([1, 0], [2, 2])
        self.assertAlmostEqual(scores[METRICS[0]], 2 ** -0.5)
        self.assertEqual(scores[METRICS[1]], 2)
        self.assertAlmostEqual(scores[METRICS[2]], 5 ** 0.5)
        for metric, best in zip(METRICS, ["A", "B", "A"]):
            self.assertEqual(retrieve(self.chunks, [[1, 0], [2, 2]], [1, 0], 1, metric)[0]["id"], best)

    def test_k_values_fields_and_context(self):
        chunks = [Chunk(str(i), f"evidence {i}", {"source": "test"}) for i in range(12)]
        vectors = [[1, i / 10] for i in range(12)]
        previous = []
        for k in [1, 3, 5, 10]:
            rows = retrieve(chunks, vectors, [1, 0], k)
            self.assertEqual(len(rows), k)
            self.assertEqual(rows[:len(previous)], previous)
            self.assertEqual([r["rank"] for r in rows], list(range(1, k + 1)))
            self.assertTrue({"rank", "id", "text", "score", "metadata", "scores"} <= rows[0].keys())
            previous = rows
        self.assertEqual(len(retrieve(self.chunks, [[1, 0], [1, 1]], [1, 0], 10)), 2)

    def test_chunk_boundaries(self):
        chunks = chunk_documents([{"id": "D", "text": "a b c d e f g", "metadata": {"title": "T"}}], size=4, overlap=1)
        self.assertEqual([c.text for c in chunks], ["a b c d", "d e f g"])
        self.assertEqual(chunks[1].metadata["document_id"], "D")
        self.assertEqual(chunks[1].metadata["title"], "T")

    def test_invalid_vectors(self):
        for a, b in [([], []), ([1], [1, 2]), ([0, 0], [1, 0]), ([float("nan")], [1])]:
            with self.assertRaises(ValueError):
                vector_scores(a, b)

    def test_live_request_uses_only_selected_context(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Answer [A]"))])
        rows = retrieve(self.chunks, [[1, 0], [0, 1]], [1, 0], 1)
        self.assertEqual(answer_question(client, "demo-model", "question", rows), "Answer [A]")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "demo-model")
        prompt = request["messages"][1]["content"]
        self.assertIn(build_context(rows), prompt)
        self.assertNotIn("second evidence", prompt)


if __name__ == "__main__":
    unittest.main()
