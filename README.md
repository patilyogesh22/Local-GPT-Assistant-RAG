Local-GPT-Assistant-RAG

Overview of Approach

This project implements a Retrieval-Augmented Generation (RAG) system that answers user questions strictly using information from uploaded documents.
The workflow:

User uploads PDF/TXT/DOCX/CSV files.

Text is extracted and split into manageable chunks (~400 words).

SentenceTransformers generates embeddings for each chunk.

FAISS builds a vector index for fast similarity search.

When a question is asked, the system retrieves the most relevant chunks.

A local Hugging Face model (FLAN-T5) generates an answer using only the retrieved context.

If the answer is not present in the documents, the system responds:
“I don’t have enough information in the uploaded documents.”

Libraries Used
Purpose	Library
UI	Streamlit
Embeddings	SentenceTransformers (all-mpnet-base-v2)
Vector Search	FAISS
Local LLM	Hugging Face Transformers (google/flan-t5-small)
Text Extraction	PyPDF, python-docx, pandas

xample Input & Output
Uploaded Document (example excerpt):
Profile Summary:
Detail-oriented Computer Science student skilled in Python, ML, and full-stack development.
Experienced in building AI-powered applications and real-world projects.

User Question:
What is the profile summary?

System Output:
The profile summary states that you are a detail-oriented Computer Science student skilled in Python,
machine learning, and full-stack development, with experience in building AI-powered applications.

Referenced Source Chunks:
Resume_Yp.pdf - Chunk 0

How Out-of-Scope Queries Are Handled

The system strictly avoids hallucination using two layers of safety:

1. Lexical Overlap Check

Before generating an answer, the system checks whether the question shares meaningful words with the retrieved document chunks.
If the question (e.g., “Who is Virat Kohli?”) contains zero overlap with the document content, the model is not allowed to answer.

2. Forced Fallback Response

If context is missing or irrelevant, the system must return:

"I don’t have enough information in the uploaded documents."


This prevents the model from guessing or inventing incorrect answers.
