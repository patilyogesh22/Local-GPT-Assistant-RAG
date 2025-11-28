Project Name: Smart Document QA Assistant

Purpose:
The purpose of this project is to build a Retrieval-Augmented Generation (RAG) system
that can answer user questions by using only the content of uploaded documents.
The system must avoid hallucinations and clearly say when the answer is not present
in the data.

Features:
1. Users can upload multiple documents in formats like PDF, TXT, CSV, and DOCX.
2. The system splits documents into smaller chunks and creates vector embeddings.
3. A vector index is used to retrieve the most relevant chunks for a user query.
4. A large language model then generates answers only from the retrieved chunks.
5. When no relevant chunks are found, the assistant responds:
   "I don’t have enough information in the uploaded documents."

Tech Stack:
- Python, Streamlit for the user interface.
- SentenceTransformers for embedding generation.
- FAISS for vector similarity search.
- OpenAI model for final answer generation.

Use Cases:
- Quickly understand long technical reports.
- Answer questions from project documentation.
- Summarize internal knowledge bases without exposing external data.
