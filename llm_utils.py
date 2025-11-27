from typing import List, Dict, Set
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)

FALLBACK_ANSWER = "I don’t have enough information in the uploaded documents."

SYSTEM_INSTRUCTION = """You are a question answering assistant.
You MUST strictly answer ONLY using the given context.
If the answer is not clearly present in the context, reply exactly:
"I don’t have enough information in the uploaded documents."
Do NOT use any outside knowledge.
"""


def _get_content_words(text: str) -> Set[str]:
    """Return a set of 'content words' from text (lowercased, no tiny words)."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    stop = {"the", "is", "are", "was", "were", "and", "or", "of", "a", "an", "to", "for", "in", "on", "at", "this", "that", "it", "with", "as", "by", "be", "what", "who", "how", "why", "which"}
    return {w for w in words if len(w) > 2 and w not in stop}


def build_context(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[Source: {c['file_name']} | Chunk {c['chunk_id']}]\n{c['text']}\n"
        )
    full_context = "\n\n".join(parts)

    max_chars = 2500
    if len(full_context) > max_chars:
        full_context = full_context[:max_chars]

    return full_context


def answer_from_context(question: str, chunks: List[Dict]) -> str:
    """Generate an answer using a local Hugging Face model with simple safety."""
    if not chunks:
        return FALLBACK_ANSWER

    context = build_context(chunks)

    # --- SIMPLE OUT-OF-SCOPE CHECK (lexical overlap) ---
    q_words = _get_content_words(question)
    c_words = _get_content_words(context)

    if not q_words.intersection(c_words):
        # e.g., "virat", "kohli" not in doc at all
        return FALLBACK_ANSWER

    # --- Ask model only if question appears related to context ---
    prompt = (
        SYSTEM_INSTRUCTION
        + "\n\nContext:\n"
        + context
        + "\n\nQuestion: "
        + question
        + "\n\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer or FALLBACK_ANSWER
