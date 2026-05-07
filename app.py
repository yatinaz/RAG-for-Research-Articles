"""
app.py - Streamlit research QA interface for the RAG pipeline.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Research Paper QA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

EVAL_RESULTS_FILE = Path("eval_results.json")


# ── Cached resource loaders ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def _load_collection():
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection("rag_assignment")


def _retrieve(query: str, top_k: int) -> list[dict]:
    model = _load_model()
    collection = _load_collection()

    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict] = []
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        chunks.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": doc,
                **meta,
                "similarity_score": round(1.0 - dist, 4),
            }
        )
    return chunks


def _generate(query: str, chunks: list[dict]) -> str:
    from generator import generate
    return generate(query, chunks)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("RAG Research Tool")
    st.caption("Retrieval-augmented QA over local scientific papers")
    st.divider()

    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("Retrieval Eval")

    if EVAL_RESULTS_FILE.exists():
        with EVAL_RESULTS_FILE.open(encoding="utf-8") as _f:
            _eval = json.load(_f)
        col_a, col_b = st.columns(2)
        col_a.metric("P@3", f"{_eval['precision_at_3']:.3f}")
        col_b.metric("P@5", f"{_eval['precision_at_5']:.3f}")
        st.caption(f"n={_eval['total_questions']} questions")

        if "by_type" in _eval:
            with st.expander("By question type"):
                for qtype, scores in _eval["by_type"].items():
                    st.markdown(
                        f"**{qtype}** — P@3 {scores['precision_at_3']:.3f} / "
                        f"P@5 {scores['precision_at_5']:.3f}"
                    )
    else:
        st.caption("No eval results yet.")
        st.caption("Run: `python eval.py --run-eval`")

    st.divider()
    st.caption("Stack: sentence-transformers · ChromaDB · Groq · Streamlit")


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("Research Paper QA")

with st.form("search_form"):
    query = st.text_input(
        "Query",
        placeholder="e.g. How does inhibition of mitochondrial respiration prevent melanoma brain metastasis?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Search", type="primary", use_container_width=False)

if not submitted or not query.strip():
    st.stop()

# ── Retrieval ─────────────────────────────────────────────────────────────────

with st.spinner("Retrieving relevant passages..."):
    try:
        chunks = _retrieve(query, top_k)
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        st.stop()

# ── Generation ────────────────────────────────────────────────────────────────

with st.spinner("Generating answer via Groq..."):
    try:
        answer = _generate(query, chunks)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
        st.stop()

# ── Answer display ────────────────────────────────────────────────────────────

st.subheader("Answer")
st.markdown(answer)

st.divider()
st.subheader(f"Sources — {len(chunks)} retrieved chunks")

for i, chunk in enumerate(chunks, 1):
    page = (
        f"p.{chunk['page_start']}"
        if chunk["page_start"] == chunk["page_end"]
        else f"pp.{chunk['page_start']}-{chunk['page_end']}"
    )
    score = chunk.get("similarity_score", 0.0)
    ctype = chunk.get("chunk_type", "text")

    label = (
        f"[{score:.3f}]  {chunk['source_file']}  —  "
        f"{chunk['section']}  —  {page}"
    )

    with st.expander(label, expanded=(i == 1)):
        meta_cols = st.columns(4)
        meta_cols[0].metric("Similarity", f"{score:.3f}")
        meta_cols[1].metric("Type", ctype)
        meta_cols[2].metric("Tokens", chunk.get("token_count", "—"))
        meta_cols[3].metric("Pages", page)

        st.markdown(f"**File:** `{chunk['source_file']}`")
        st.markdown(f"**Section:** {chunk['section']}")

        preview = chunk["text"]
        if len(preview) > 900:
            preview = preview[:900] + "…"
        st.text(preview)
