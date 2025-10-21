# evaluation/build_known_from_faiss.py
"""
Generate evaluation/question_sets/known_questions.jsonl
directly from your FAISS retriever's indexed docs.

This version handles 'Question:' / 'Answer:' format.
"""

import os, json, random
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===== Config =====
INDEX_PATH = "retriever/faiss_index"
OUT_PATH   = "evaluation/question_sets/known_questions.jsonl"
N_SAMPLES  = 10   # 0 = all
random.seed(42)

# ===== Load FAISS retriever =====
print(f"📂 Loading FAISS index from {INDEX_PATH} ...")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

docs = list(vs.docstore._dict.values())
print(f"📄 Loaded {len(docs)} documents from FAISS index.")

# ===== Extract question-answer pairs =====
qa_pairs = []
for d in docs:
    text = d.page_content.strip()
    if text.lower().startswith("question:"):
        parts = text.split("Answer:", 1)
        if len(parts) == 2:
            q = parts[0].replace("Question:", "").strip()
            a = parts[1].strip()
            if q and a:
                qa_pairs.append({
                    "question": q,
                    "reference_answer": a,
                    "source": d.metadata.get("source", "")
                })

print(f"🧩 Extracted {len(qa_pairs)} QA pairs.")

# ===== Random sample =====
if N_SAMPLES and len(qa_pairs) > N_SAMPLES:
    qa_pairs = random.sample(qa_pairs, N_SAMPLES)

# ===== Write to JSONL =====
out_path = Path(OUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    for qa in qa_pairs:
        f.write(json.dumps(qa, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(qa_pairs)} questions → {out_path}")
