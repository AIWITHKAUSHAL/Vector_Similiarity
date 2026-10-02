"""Export real embedding results for all three metrics and K = 1, 3, 5, 10."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import (
    K_VALUES,
    METRICS,
    build_context,
    chunk_documents,
    embed_texts,
    retrieve,
)

BASE_URL = "https://api.euron.one/api/v1/euri"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query", default="How does changing K affect the context supplied to an LLM?"
    )
    parser.add_argument("--output", default="experiment_results.json")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("The query must not be empty.")
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    api_key = os.getenv("EURI_API_KEY", "")
    if not api_key:
        parser.error("Set EURI_API_KEY in .env or the environment to create embeddings.")
    chunks = chunk_documents(
        json.loads((root / "data" / "documents.json").read_text(encoding="utf-8"))
    )
    model_name = os.getenv("EURI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
    with OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60, max_retries=1) as client:
        vectors = embed_texts(client, model_name, [c.text for c in chunks])
        query_vector = embed_texts(client, model_name, [args.query.strip()])[0]
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
