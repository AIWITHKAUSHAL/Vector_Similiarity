# Vector Search and Similarity Analysis

A Streamlit retrieval lab with local semantic embeddings and live, context-grounded answers through EURI. Includes a 14-document educational corpus, custom text uploads, exact Top-K search, three scoring methods, and side-by-side K experiments.

## Run

Python 3.11–3.13 recommended. From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set your EURI_API_KEY.
.\.venv\Scripts\python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The first search downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face; this requires internet access and disk space for the model and dependencies. Later searches reuse the cached model. Embeddings run locally; no embedding API key is required.

The `.env` file is ignored by Git. Alternatively, enter your key in the app's password field. Without an API key, retrieval and comparison still work. Generation requires a working EURI key, account access to the selected model, and network access. Requests may incur provider charges.

EURI integration uses the supplied OpenAI-compatible configuration:

```python
client = OpenAI(
    api_key=os.environ["EURI_API_KEY"],
    base_url="https://api.euron.one/api/v1/euri",
)
```

The default chat model is `gemini-3.5-flash-lite`. Set `EURI_MODEL` or edit the model field if your account uses another model. This chat model generates answers; the separate local embedding model produces the vectors.

## Demonstrate the assignment

1. Keep the sample dataset and search: **How does changing K affect the context supplied to an LLM?**
2. Inspect the result table and expand a row: rank, chunk ID, complete text, vector score, and metadata are visible. Scores are calculated, not hardcoded.
3. Open **K & metric experiments**. It shows K = 1, 3, 5, 10 for the same query, with actual result counts, ordered IDs, context word counts, and character counts. Export the results if desired.
4. Compare cosine similarity, dot product, and Euclidean distance. Change the ranking metric in the sidebar and retrieve again to see the corresponding evidence.
5. Choose K = 1, retrieve, inspect **Exact context supplied to the LLM**, then click **Generate with EURI**. Repeat with K = 3, 5, and 10. Each live call receives only the selected chunks; it does not silently include the whole corpus or previous answers.
6. Compare answer coverage and citations across K. More chunks can add useful support but also irrelevant text. The app reports word/character counts, not fabricated API token counts.
7. Optionally upload a UTF-8 `.txt` document. It replaces the sample corpus and is chunked into at most 110 words with 25-word overlap. If fewer than K chunks exist, all available chunks are returned.

## How retrieval works

### Interactive architecture visualizer

Open [docs/architecture_visualizer.html](docs/architecture_visualizer.html) directly in a browser. This standalone, offline page includes a clickable architecture diagram, play/pause and step controls, three walkthrough scenarios, a Python source inspector, and all 12 saved metric/K experiments. Tables, context previews, charts, and JSON exports use `experiment_results.json`. Live generation remains in the Streamlit app.

After changing the Python source or regenerating the experiment report, refresh the embedded snapshots:

```powershell
python docs/build_visualizer.py
```


`documents → chunks + metadata → local embeddings → query embedding → exact scores → sorted Top-K → EURI prompt → cited answer`

Both documents and queries use the same 384-dimensional MiniLM model. Search scores every chunk and uses stable chunk IDs to break ties. It ranks similarity in descending order and distance in ascending order. The app does not apply additional vector normalization (`normalize_embeddings=False`); model-internal normalization may still make rankings coincide.

| Concept | Formula | Interpretation |
| --- | --- | --- |
| Cosine similarity | `(q · d) / (‖q‖ ‖d‖)` | Higher is closer in direction. Range −1 to 1. |
| Dot product | `Σ qᵢdᵢ` | Higher is stronger alignment, influenced by magnitude. No fixed range. |
| Euclidean distance | `√Σ(qᵢ − dᵢ)²` | Lower is closer; zero means identical vectors. |

Scores are not confidence percentages or proof of relevance. A search always returns up to K candidates even when evidence is weak. No relevance cutoff, metadata filtering, or reranker is implemented.

For query `[1, 0]`, candidate A `[1, 0]` has cosine 1, dot product 1, and distance 0. Candidate B `[2, 2]` has cosine about 0.707, dot product 2, and distance about 2.236. Thus cosine and distance favor A, while the raw dot product favors B. For unit vectors dot product equals cosine, and squared Euclidean distance equals `2 − 2 × cosine`, giving equivalent rankings.

K changes the amount of evidence supplied to the LLM, not the embedding dimension or the scores for individual chunks. For a fixed query and metric, smaller result lists are prefixes of larger ones. Larger K can improve evidence coverage but can increase input tokens, cost, latency, and distraction. It does not guarantee better answers. The generation prompt requests citations and acknowledgement of insufficient evidence; model compliance and citation correctness are not guaranteed and should be checked.

## Files and validation

- `app.py`: interactive UI, embedding cache, experiments, live EURI integration.
- `retrieval.py`: chunking, scoring, ranking, context assembly, generation request.
- `data/documents.json`: authored sample corpus and metadata.
- `tests/test_retrieval.py`: numerical examples, K behavior, chunk boundaries, invalid vectors, and prompt isolation using a mock client.
- `run_experiments.py`: command-line export of real results for all 12 metric/K combinations.
- `VIDEO_SCRIPT.md`: demonstration and explanation outline.

```powershell
python -m unittest discover -s tests -v
```

To save a reproducible experiment report with every result field:

```powershell
.\.venv\Scripts\python run_experiments.py --output experiment_results.json
```

The included `experiment_results.json` was generated with real MiniLM embeddings for the default question. All three metrics ranked `DOC-08:chunk-1` first in this run. The retrieved context sizes were:

| K | Retrieved chunks | Context words (including chunk ID labels) |
| --- | --- | --- |
| 1 | 1 | 62 |
| 3 | 3 | 177 |
| 5 | 5 | 284 |
| 10 | 10 | 543 |

These are observed retrieval results, not a live LLM evaluation. Regenerate the report to experiment with another query.

Core tests use only the Python standard library. They do not call a paid API or download an embedding model. To validate a real completion, run the app with your EURI key and click Generate; a mock test does not establish provider availability.

## GitHub and YouTube submission

Create a GitHub repository and push the complete project (exclude `.env`, `.venv`, and model caches):

```powershell
git init
git add .
git commit -m "Build vector search and live EURI retrieval demo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Record the walkthrough in `VIDEO_SCRIPT.md`, upload it to YouTube with visibility your reviewer can access, and submit both links. Check that the repository is accessible and the video demonstrates the same code. Hide your API key and `.env` while recording.
