"""Export real embedding results for all three metrics and K = 1, 3, 5, 10."""

import argparse
import json
from pathlib import Path

from retrieval import K_VALUES, METRICS, build_context, chunk_documents, retrieve


def main():
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query", default="How does changing K affect the context supplied to an LLM?"
    )
    parser.add_argument("--output", default="experiment_results.json")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("The query must not be empty.")
    root = Path(__file__).resolve().parent
    chunks = chunk_documents(
        json.loads((root / "data" / "documents.json").read_text(encoding="utf-8"))
    )
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(
        model_name, cache_folder=str(root / ".cache" / "models")
    )
    vectors = model.encode(
        [c.text for c in chunks], normalize_embeddings=False
    ).tolist()
    query_vector = model.encode(args.query.strip(), normalize_embeddings=False).tolist()
    experiments = []
    for metric in METRICS:
        for k in K_VALUES:
            results = retrieve(chunks, vectors, query_vector, k, metric)
            context = build_context(results)
            experiments.append(
                {
                    "metric": metric,
                    "k": k,
                    "actual_chunks": len(results),
                    "context_words": len(context.split()),
                    "context_characters": len(context),
                    "results": results,
                }
            )
            print(
                f"{metric:20s} K={k:2d} chunks={len(results):2d} words={len(context.split()):4d} best={results[0]['id']}"
            )
    report = {
        "query": args.query.strip(),
        "embedding_model": model_name,
        "dimensions": len(query_vector),
        "experiments": experiments,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
