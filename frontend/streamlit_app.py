import streamlit as st
import requests
import json

# ── Config ───────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"  # Change to your deployed URL when hosted

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide"
)

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Document Q&A")
    st.markdown("---")

    # Upload section
    st.subheader("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("Extracting text, creating embeddings..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success("✅ Document processed!")
                        stats = data["stats"]
                        st.info(
                            f"📊 **Stats**\n"
                            f"- Chunks created: {stats['num_chunks']}\n"
                            f"- Chunk size: {stats['chunk_size']} chars\n"
                            f"- Time: {stats['ingestion_time_seconds']}s"
                        )
                        st.session_state.document_loaded = True
                        st.session_state.chat_history = []
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")

    st.markdown("---")

    # Advanced settings
    st.subheader("⚙️ Settings")
    top_k = st.slider("Top K chunks to retrieve", 1, 8, 3)
    chunk_size = st.selectbox("Chunk size", [256, 500, 1024], index=1)

    if st.button("Apply Settings"):
        try:
            requests.post(
                f"{API_URL}/configure",
                params={"chunk_size": chunk_size, "top_k": top_k}
            )
            st.success("Settings applied!")
        except:
            st.warning("Could not apply settings.")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("💬 Ask Questions About Your Document")

if not st.session_state.document_loaded:
    st.info("👈 Upload a PDF from the sidebar to get started.")

    # Example use cases
    st.markdown("### What you can do:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("📑 **Research Papers**\nAsk about methodology, findings, conclusions")
    with col2:
        st.markdown("📋 **Legal Documents**\nUnderstand clauses, obligations, terms")
    with col3:
        st.markdown("📘 **Technical Docs**\nGet answers from manuals, reports, specs")

else:
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            with st.expander(f"📚 Source chunks used (relevance scores)"):
                for i, (chunk, score) in enumerate(zip(chat["sources"], chat["scores"])):
                    st.markdown(f"**Chunk {i+1}** — Score: `{score:.3f}`")
                    st.markdown(f"> {chunk[:400]}...")
                    st.markdown("---")

    # Question input
    question = st.chat_input("Ask a question about your document...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.write(data["answer"])

                        # Metadata
                        col1, col2 = st.columns(2)
                        col1.caption(f"⏱️ Response time: {data['response_time_seconds']}s")
                        col2.caption(f"🔢 Tokens used: {data['tokens_used']}")

                        with st.expander("📚 Source chunks used"):
                            for i, (chunk, score) in enumerate(zip(data["sources"], data["scores"])):
                                st.markdown(f"**Chunk {i+1}** — Relevance: `{score:.3f}`")
                                st.markdown(f"> {chunk[:400]}...")
                                st.markdown("---")

                        # Save to history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": data["answer"],
                            "sources": data["sources"],
                            "scores": data["scores"]
                        })

                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Could not reach API: {e}")
                    st.info("Make sure the FastAPI backend is running on localhost:8000")
