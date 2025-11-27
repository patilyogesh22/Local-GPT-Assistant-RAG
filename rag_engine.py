import os
import pickle
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx
import pandas as pd

CHUNK_SIZE = 400  # words
INDEX_DIR = "data/index"
os.makedirs(INDEX_DIR, exist_ok=True)


class RAGEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.dim = self.embed_model.get_sentence_embedding_dimension()

    # ---------- File loading ----------
    def load_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text

        elif ext == ".docx":
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return df.to_string(index=False)

        else:
            return ""

    # ---------- Chunking ----------
    def chunk_text(self, text: str, file_name: str) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE):
            chunk_words = words[i:i + CHUNK_SIZE]
            if not chunk_words:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append(
                {
                    "text": chunk_text,
                    "file_name": file_name,
                    "chunk_id": len(chunks),
                }
            )
        return chunks

    # ---------- Add & index documents ----------
    def add_documents(self, file_paths: List[str]):
        new_chunks = []
        for path in file_paths:
            text = self.load_file(path)
            if not text or not text.strip():
                continue
            file_name = os.path.basename(path)
            chunks = self.chunk_text(text, file_name)
            new_chunks.extend(chunks)

        if not new_chunks:
            return

        texts = [c["text"] for c in new_chunks]
        embeddings = self.embed_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        ).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []

        self.index.add(embeddings)
        self.metadata.extend(new_chunks)
        self.save_index()

    # ---------- Save / load index ----------
    def save_index(self):
        if self.index is None:
            return
        index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
        meta_path = os.path.join(INDEX_DIR, "metadata.pkl")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
        meta_path = os.path.join(INDEX_DIR, "metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

    # ---------- Retrieval ----------
    def retrieve(self, query: str, top_k: int = 5):
        """Always return up to top_k most similar chunks (no threshold)."""
        if self.index is None or not self.metadata:
            return []

        top_k = min(top_k, len(self.metadata))

        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        seen = set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if idx in seen:
                continue
            seen.add(idx)

            score = 1 / (1 + dist)  # simple similarity
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            results.append(meta)

        return results
