# Video walkthrough (approximately 5–7 minutes)

## 0:00 — Goal and application

“This project retrieves Top-K document chunks for a user query and uses the retrieved text as context for a live language model. Every result displays its rank, document/chunk ID, text, numerical score, and metadata.”

Show the app and the sample corpus. Explain that the notes are authored demo data. Show `data/documents.json` and its IDs, texts, and metadata.

## 0:40 — Embeddings and chunking

“An embedding represents text as a numerical vector. Related meanings tend to have nearby embeddings. I use the same local MiniLM model for document chunks and the query, producing 384-dimensional vectors. The embedding model is separate from the EURI chat model.”

Show `chunk_documents` in `retrieval.py` and `get_encoder` / `embed_documents` in `app.py`. Explain the 110-word chunk size, 25-word overlap, source metadata, and local caching.

## 1:30 — Similarity and distance

Open **How scores work** after a search. Explain cosine's direction-based comparison and range of −1 to 1; larger is better. Explain dot product's dependence on magnitude. Explain Euclidean straight line distance; smaller is better.

Use the worked example q = [1, 0], A = [1, 0], B = [2, 2] to show that dot product can rank differently. Mention that unit vectors make these rankings equivalent. “A cosine of 0.8 is not an 80% probability of correctness.”

Show `vector_scores` and the ascending/descending sort in `retrieve`.

## 2:30 — Top-K retrieval and required output

Search **How does changing K affect the context supplied to an LLM?**

Expand a result and identify every required field. Explain that exact search scores every chunk, sorts by the selected metric, and returns up to K. If fewer chunks exist, it returns the available chunks.

Open **K & metric experiments**. Show all four rows for K = 1, 3, 5, and 10. Read the actual observed IDs and context sizes from your run. Compare metric rankings; if they coincide, explain the normalization relationship rather than claiming they differ.

## 3:30 — Live LLM and context comparison

Set K = 1, retrieve, and open the exact context panel. Click Generate and show the live answer and citation. Repeat at K = 3, 5, and 10, showing each context and live answer. Pause the recording during network waits if necessary. Never show your key.

“The same question gets different amounts of evidence. One chunk is focused but can miss details. Three or five can add useful supporting facts. Ten may cover more material but can also introduce noise. More input generally means more tokens, and possibly higher cost and latency. More context does not guarantee better output.”

Compare what the real answers include and omit; do not invent differences. The table measures words and characters, not provider token counts.

## 5:00 — Code and verification

Show `answer_question`, the supplied EURI base URL, model name, messages, and selected context construction. Explain the instruction to cite chunks and acknowledge missing evidence. Note that citations still require checking.

Run `python -m unittest discover -s tests -v`. Explain that these tests verify ranking mathematics, K sizes, and selected-context isolation, while the UI demonstration verifies the live API call.

## 6:00 — Submission

Show the GitHub repository with complete code and README. Confirm that `.env` is excluded. Submit the repository link and the accessible YouTube link through the assignment portal.
