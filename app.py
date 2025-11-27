# app.py
import os
import streamlit as st

from rag_engine import RAGEngine
from llm_utils import answer_from_context

UPLOAD_DIR = "data/uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Local GPT Assistant", page_icon="🤖")

st.title("📄 Local GPT Assistant (RAG)")
st.write("Upload documents and ask questions. Answers are based **only** on uploaded files.")

# Initialize engine in session
if "rag_engine" not in st.session_state:
    engine = RAGEngine()
    engine.load_index()
    st.session_state["rag_engine"] = engine

engine: RAGEngine = st.session_state["rag_engine"]

# --- File upload ---
st.header("1️⃣ Upload Documents")
uploaded_files = st.file_uploader(
    "Upload .txt, .pdf, .csv, .docx files",
    type=["txt", "pdf", "csv", "docx"],
    accept_multiple_files=True
)

import shutil

def safe_delete_folder(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except:
            pass
        try:
            os.makedirs(path, exist_ok=True)
        except:
            pass


if uploaded_files and st.button("Process & Index Files"):
    UPLOAD_DIR = "data/uploaded"
    INDEX_DIR = "data/index"

    # 1️⃣ UNLOAD OLD ENGINE BEFORE DELETING INDEX
    st.session_state["rag_engine"] = None

    # 2️⃣ DELETE FOLDERS SAFELY
    safe_delete_folder(INDEX_DIR)
    safe_delete_folder(UPLOAD_DIR)

    # 3️⃣ CREATE UPLOAD FOLDER AGAIN
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 4️⃣ SAVE NEW FILES
    saved_paths = []
    for up in uploaded_files:
        save_path = os.path.join(UPLOAD_DIR, up.name)
        with open(save_path, "wb") as f:
            f.write(up.read())
        saved_paths.append(save_path)

    # 5️⃣ RECREATE FRESH ENGINE
    engine = RAGEngine()
    st.session_state["rag_engine"] = engine

    # 6️⃣ REBUILD INDEX
    with st.spinner("Indexing documents..."):
        engine.add_documents(saved_paths)

    st.success("🎉 Documents processed and index rebuilt successfully!")


# --- QA section ---
st.header("2️⃣ Ask Questions")

question = st.text_input("Your question:", placeholder="e.g., What is the purpose of this project?")
top_k = st.slider("Number of source chunks:", 1, 10, 5)

if st.button("Get Answer") and question.strip():
    with st.spinner("Thinking..."):
        chunks = engine.retrieve(question, top_k=top_k)
        answer = answer_from_context(question, chunks)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Referenced Sources")
    if chunks:
        for c in chunks:
            with st.expander(f"{c['file_name']} - Chunk {c['chunk_id']} (score: {c['score']:.2f})"):
                st.write(c["text"])
    else:
        st.write("No relevant context found. (Similarity below threshold)")
