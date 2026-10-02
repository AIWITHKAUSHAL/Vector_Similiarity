"""Run with: python -m streamlit run app.py"""

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from retrieval import (
    K_VALUES,
    METRICS,
    answer_question,
    build_context,
    chunk_documents,
    embed_texts,
    retrieve,
)

load_dotenv()
ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = os.getenv("EURI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
BASE_URL = "https://api.euron.one/api/v1/euri"

st.set_page_config(page_title="Vector Search Lab", page_icon="🔎", layout="wide")


def embed_with_euri(api_key, texts):
    with OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60, max_retries=1) as client:
        return embed_texts(client, EMBEDDING_MODEL, texts)


@st.cache_data(show_spinner=False)
def embed_documents(texts, _api_key):
    return embed_with_euri(_api_key, texts)


st.title("Vector Search Lab")
st.caption(
    "Explore semantic retrieval • Compare vector scores • Ground a live LLM in your Top-K results"
)

with st.sidebar:
    st.header("Search settings")
    metric = st.selectbox("Ranking metric", METRICS)
    k = st.select_slider("Chunks supplied to the LLM (K)", options=K_VALUES, value=3)
    st.divider()
    st.subheader("Live EURI connection")
    api_key = st.text_input(
        "EURI API key", type="password", help="Or set EURI_API_KEY in your .env file."
    ) or os.getenv("EURI_API_KEY", "")
    model = st.text_input(
        "Chat model", value=os.getenv("EURI_MODEL", "gemini-3.5-flash-lite")
    )
    st.caption(
        "Chunks and your query are sent to EURI for embedding when you click Retrieve, and to the chat model when you click Generate. Keep your key out of recordings."
    )
    uploaded = st.file_uploader("Optional custom document (.txt, UTF-8)", type=["txt"])
    st.caption(
        "The sample corpus contains 14 documents. An upload replaces it and is split into overlapping chunks."
    )

documents = json.loads((ROOT / "data" / "documents.json").read_text(encoding="utf-8"))
if uploaded is not None:
    try:
        uploaded_text = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("Please upload a UTF-8 text file.")
        st.stop()
    if len(uploaded_text) > 100_000:
        st.error("Please use a document smaller than 100,000 characters for this demo.")
        st.stop()
    documents = [
        {
            "id": "UPLOAD",
            "text": uploaded_text,
            "metadata": {
                "source": uploaded.name,
                "title": uploaded.name,
                "topic": "custom",
            },
        }
    ]
chunks = chunk_documents(documents)
if not chunks:
    st.warning("The document has no text. Upload a nonempty file.")
    st.stop()

query = st.text_input(
    "Ask a question", value="How does changing K affect the context supplied to an LLM?"
)
search_signature = (
    query.strip(),
    metric,
    k,
    tuple(c.text for c in chunks),
    tuple(c.id for c in chunks),
    tuple(str(c.metadata) for c in chunks),
)
if st.session_state.get("search_signature") != search_signature:
    st.session_state.pop("search", None)
    st.session_state.pop("answer", None)

if st.button("Retrieve and compare", type="primary"):
    if not query.strip():
        st.warning("Enter a question first.")
    elif not api_key:
        st.warning("Add EURI_API_KEY to .env or enter it in the sidebar to create embeddings.")
    else:
        try:
            with st.spinner(
                f"Creating embeddings with {EMBEDDING_MODEL} and ranking chunks…"
            ):
                vectors = embed_documents(tuple(c.text for c in chunks), api_key)
                query_vector = embed_with_euri(api_key, (query.strip(),))[0]
                rankings = {
                    m: retrieve(chunks, vectors, query_vector, 10, m) for m in METRICS
                }
            st.session_state.search = {
                "rankings": rankings,
                "dimension": len(query_vector),
            }
            st.session_state.search_signature = search_signature
            st.session_state.pop("answer", None)
        except Exception as exc:  # noqa: BLE001 - show a friendly UI error instead of a traceback
            st.error(
                f"Embedding creation failed. Check your EURI API key, the {EMBEDDING_MODEL} model, quota, and connection. Retrieval cannot run without embeddings."
                f"\n\nDetails: {type(exc).__name__}: {str(exc)[:300]}"
            )

search = st.session_state.get("search")
if search:
    results = search["rankings"][metric][:k]
    a, b, c = st.columns(3)
    a.metric("Available chunks", len(chunks))
    b.metric("Embedding dimensions", search["dimension"])
    c.metric("Retrieved / requested", f"{len(results)} / {k}")
    results_tab, experiment_tab, concepts_tab = st.tabs(
        ["Retrieved results", "K & metric experiments", "How scores work"]
    )
    with results_tab:
        st.caption(
            f"Ranked by {metric}. "
            + ("Lower is better." if metric == METRICS[2] else "Higher is better.")
        )
        st.dataframe(
            [
                {
                    "Rank": r["rank"],
                    "Document/Chunk ID": r["id"],
                    "Text": r["text"],
                    "Vector score": r["score"],
                    "Metadata": json.dumps(r["metadata"]),
                }
                for r in results
            ],
            hide_index=True,
            width="stretch",
        )
        for r in results:
            with st.expander(f"#{r['rank']} · {r['id']} · {metric}: {r['score']:.6f}"):
                st.write(r["text"])
                st.json(r["metadata"])
                st.write({name: round(value, 6) for name, value in r["scores"].items()})
        st.download_button(
            "Download retrieved results (JSON)",
            json.dumps(results, indent=2),
            "results.json",
            "application/json",
        )
        with st.expander("Exact context supplied to the LLM"):
            st.code(build_context(results), language=None)
        st.subheader("Live answer grounded in these results")
        if st.button(
            "Generate with EURI", disabled=not bool(api_key and model.strip())
        ):
            try:
                with (
                    st.spinner("Requesting a live answer from EURI…"),
                    OpenAI(
                        api_key=api_key, base_url=BASE_URL, timeout=60, max_retries=1
                    ) as client,
                ):
                    answer = answer_question(
                        client, model.strip(), query.strip(), results
                    )
                st.session_state.answer = {"text": answer, "model": model.strip()}
            except Exception as exc:  # noqa: BLE001 - show a friendly UI error instead of a traceback
                st.error(
                    "EURI could not generate an answer. Check your API key, model availability, account quota, and connection. Your retrieval results are still available."
                    f"\n\nDetails: {type(exc).__name__}: {str(exc)[:300]}"
                )
        if not api_key:
            st.info(
                "Add EURI_API_KEY to .env or enter it in the sidebar to enable live generation."
            )
        if "answer" in st.session_state:
            st.caption(
                f"Generated live by {st.session_state.answer['model']} with K = {k}"
            )
            st.markdown(st.session_state.answer["text"])
    with experiment_tab:
        experiment = []
        for amount in K_VALUES:
            subset = search["rankings"][metric][:amount]
            context = build_context(subset)
            experiment.append(
                {
                    "Requested K": amount,
                    "Actual chunks": len(subset),
                    "Context words": len(context.split()),
                    "Context characters": len(context),
                    "Chunk IDs in rank order": ", ".join(r["id"] for r in subset),
                }
            )
        st.subheader("Same query, four context sizes")
        st.dataframe(experiment, hide_index=True, width="stretch")
        st.bar_chart(
            {
                "K": [str(r["Requested K"]) for r in experiment],
                "Context words": [r["Context words"] for r in experiment],
            },
            x="K",
            y="Context words",
        )
        st.caption(
            "Word and character counts measure the retrieved context, not API token usage. Actual token counts depend on the provider's tokenizer."
        )
        st.write(
            "K = 1 supplies focused evidence. K = 3 and 5 can add supporting details. K = 10 provides broader context, but may include unrelated chunks and increase input tokens. More context does not guarantee a better answer."
        )
        st.subheader(f"Compare all three metrics at K = {k}")
        for name, col in zip(METRICS, st.columns(3)):
            with col:
                st.markdown(f"**{name}**")
                st.dataframe(
                    [
                        {"Rank": r["rank"], "Chunk ID": r["id"], "Score": r["score"]}
                        for r in search["rankings"][name][:k]
                    ],
                    hide_index=True,
                )
        st.download_button(
            "Download K experiment",
            json.dumps(experiment, indent=2),
            "k_experiment.json",
            "application/json",
        )
    with concepts_tab:
        st.markdown(r"""
**Cosine similarity** compares direction. Higher is better; its range is [-1, 1].
""")
        st.latex(r"\cos(q,d)=\frac{q\cdot d}{\|q\|\|d\|}")
        st.write(
            "1 means same direction, 0 means orthogonal, and −1 means opposite. A score of 0.8 does not mean 80% confidence or accuracy."
        )
        st.markdown(
            "**Dot product** uses both alignment and vector magnitude. Higher is better; it has no fixed range."
        )
        st.latex(r"q\cdot d=\sum_i q_i d_i")
        st.markdown(
            "**Euclidean distance** measures straight line separation. Lower is better; 0 means identical vectors."
        )
        st.latex(r"\|q-d\|_2=\sqrt{\sum_i(q_i-d_i)^2}")
        st.write(
            "Example: q = [1, 0], A = [1, 0], B = [2, 2]. Cosine ranks A first (1 versus 0.707); dot product ranks B first (2 versus 1); Euclidean distance ranks A first (0 versus 2.236)."
        )
        st.write(
            "For unit vectors, dot product equals cosine and squared Euclidean distance equals 2 − 2 × cosine. Their rankings then coincide. We do not request extra normalization; a model may already emit normalized vectors, so rankings may still coincide."
        )
else:
    st.info(
        "Click Retrieve and compare to inspect scores and run all four K experiments."
    )

with st.expander("Embedding and corpus details"):
    st.write(
        f"Embedding model: {EMBEDDING_MODEL}. Query and chunks use the same EURI embedding model. The chat model is used only for answer generation."
    )
    st.write(
        "Chunks contain up to 110 whitespace-separated words with 25-word overlap. Search exhaustively scores every chunk. The sample documents are authored educational notes, not external reference material."
    )
    st.json([{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks])
