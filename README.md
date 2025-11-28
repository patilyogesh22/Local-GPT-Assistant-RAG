# 🧠 Local-GPT-Assistant-RAG

The **Local-GPT-Assistant-RAG** is a Retrieval-Augmented Generation (RAG) system designed to answer user questions **only using the content of uploaded documents**.  
It ensures accuracy, avoids hallucinations, and clearly informs the user when the information is not present in the provided data.

---

## 🎯 Project Purpose

The goal of this project is to create an intelligent question-answering system that can:

- Ingest multiple documents  
- Build a searchable knowledge base  
- Retrieve only relevant information  
- Generate accurate answers using a Large Language Model  
- Avoid hallucinations and maintain trustworthiness  

If the system cannot find enough relevant context, it responds with:

> **"I don’t have enough information in the uploaded documents."**

---

## ✨ Key Features

1. **Multi-document Upload**  
   Supports PDF, TXT, CSV, and DOCX formats.

2. **Automatic Document Chunking**  
   Documents are split into manageable text chunks for efficient search.

3. **Vector Embeddings**  
   Uses SentenceTransformers to convert document chunks into semantic embeddings.

4. **Efficient Similarity Search**  
   FAISS is used as a high-performance vector index to retrieve relevant chunks.

5. **RAG-Based Answer Generation**  
   An OpenAI model generates answers strictly from retrieved document content.

6. **Hallucination Prevention**  
   If the model cannot find relevant text, it returns a safe fallback answer.

---

## 🧰 Tech Stack

| Component | Technology |
|----------|------------|
| Frontend / UI | **Streamlit** |
| Backend Logic | **Python** |
| Embedding Model | **SentenceTransformers** |
| Vector Index | **FAISS** |
| LLM for QA | **OpenAI Chat / GPT models** |
| Document Handling | PyPDF2, python-docx, pandas, text loaders |

---

## 📁 How It Works

1. User uploads one or more documents.  
2. Documents are converted into raw text and split into chunks.  
3. SentenceTransformers generates embeddings for each chunk.  
4. FAISS stores embeddings and performs similarity search.  
5. The top-k relevant chunks are sent to the OpenAI model.  
6. The model generates an answer strictly based on the retrieved chunks.  
7. If no relevant context is found, the fallback safety response is returned.

---

