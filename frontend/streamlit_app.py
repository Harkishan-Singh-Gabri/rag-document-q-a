import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Document Q&A", layout="wide")

# Session state
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of {question, answer, sources, scores}

# Sidebar: upload + config 
with st.sidebar:
    st.header("Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    with st.expander("Retrieval settings", expanded=False):
        chunk_size = st.slider("Chunk size (chars)", 200, 1500, 500, step=50)
        chunk_overlap = st.slider("Chunk overlap (chars)", 0, 200, 50, step=10)
        top_k = st.slider("Top-K chunks retrieved", 1, 10, 3)

    if uploaded_file and st.button("Ingest document", use_container_width=True):
        with st.spinner("Extracting, chunking, and embedding..."):
            try:
                requests.post(
                    f"{API_URL}/configure",
                    params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "top_k": top_k},
                    timeout=10,
                )
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                resp = requests.post(f"{API_URL}/upload", files=files, timeout=120)

                if resp.status_code == 200:
                    stats = resp.json()["stats"]
                    st.session_state.doc_loaded = True
                    st.session_state.doc_name = stats["file"]
                    st.session_state.history = []
                    st.success(f"Ingested **{stats['num_chunks']} chunks** in {stats['ingestion_time_seconds']}s")
                else:
                    st.error(resp.json().get("detail", "Upload failed."))
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is it running at " + API_URL + "?")

    if st.session_state.doc_loaded:
        st.divider()
        st.caption(f"Active document: **{st.session_state.doc_name}**")
        if st.button("Clear session", use_container_width=True):
            st.session_state.doc_loaded = False
            st.session_state.doc_name = None
            st.session_state.history = []
            st.rerun()

# Main panel 
st.title("RAG Document Q&A")

if not st.session_state.doc_loaded:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

question = st.text_input("Ask a question about the document", placeholder="e.g. What is the main conclusion?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.history.insert(0, {
                    "question": question,
                    "answer": data["answer"],
                    "sources": data["sources"],
                    "scores": data["scores"],
                    "time": data.get("response_time_seconds", 0),
                })
            else:
                st.error(resp.json().get("detail", "Query failed."))
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the API. Is it running at " + API_URL + "?")

# History (most recent first) 
for item in st.session_state.history:
    with st.container(border=True):
        st.markdown(f"**Q: {item['question']}**")
        st.write(item["answer"])
        st.caption(f"{item['time']}s")

        if item["sources"]:
            with st.expander(f"📚 {len(item['sources'])} source chunk(s) used"):
                for i, (src, score) in enumerate(zip(item["sources"], item["scores"]), 1):
                    st.markdown(f"**Chunk {i}** — relevance score: `{score:.3f}`")
                    st.text(src[:400] + ("..." if len(src) > 400 else ""))
                    st.divider()
