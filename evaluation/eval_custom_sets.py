import os, json, csv, math, re, numpy as np
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

# LangChain (community splits)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from evaluate import load as load_metric

# ---------------- Config ----------------
INDEX_PATH = "retriever/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
TOPK_LIST = [1, 3, 5]

OUT_OF_KB_PATH = "evaluation/question_sets/out_of_kb.jsonl"
INFERRED_PATH  = "evaluation/question_sets/inferred_examples.jsonl"
KNOWN_PATH      = "evaluation/question_sets/known_questions.jsonl"

OUT_OF_KB_RESULTS = "evaluation/results_out_of_kb_unanswered.csv"
INFERRED_RESULTS  = "evaluation/results_inferred_metrics.csv"
KNOWN_RESULTS   = "evaluation/results_known_metrics.csv"
DETAILS_LOG       = "evaluation/details_log.csv"

SYSTEM_INSTRUCTION = (
    "You are a strict, rule-based QA assistant. "
    "Your ONLY source of truth is the provided context below. "
    "If the context text does NOT contain the answer or is unrelated, "
    "you must respond with exactly this single word (in all caps): UNANSWERED. "
    "Do NOT guess, explain, infer, or use any world knowledge. "
    "If the question asks for something not explicitly stated in the context, respond UNANSWERED. "
    "Follow this rule 100% of the time."
)

PROMPT_TEMPLATE = """{system}

Context:
{context}

Question: {question}

Answer:"""

# ------------- Utilities -------------
def setup_rag():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    retriever = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    ).as_retriever()
    llm = Ollama(model=OLLAMA_MODEL)  # use .invoke later
    return retriever, llm

def llm_answer(llm, question: str, chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(chunks) if chunks else ""
    prompt = PROMPT_TEMPLATE.format(system=SYSTEM_INSTRUCTION, context=context, question=question)
    try:
        out = llm.invoke(prompt)  # modern LC API
        # handle possible message object or string
        text = getattr(out, "content", out)
        return (text or "").strip()
    except Exception as e:
        return f"__ERROR__: {e}"

def is_unanswered(text: str) -> bool:
    if not text or not text.strip():
        return True
    return text.strip().lower() == "unanswered"

def load_jsonl(path: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def dcg_at_k(rels: List[int]) -> float:
    # standard DCG: sum((2^rel-1)/log2(i+2))
    dcg = 0.0
    for i, r in enumerate(rels):
        dcg += (2**r - 1) / math.log2(i + 2)
    return dcg

def ndcg_at_k(rels: List[int]) -> float:
    if not rels:
        return 0.0
    dcg = dcg_at_k(rels)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal)
    return 0.0 if idcg == 0 else dcg / idcg

def relevance_binary(reference_answer: str, chunk: str) -> int:
    # simple proxy: 1 if any reference token appears in chunk; else 0
    if not reference_answer:
        return 0
    ref = reference_answer.lower().strip()
    chunk_l = chunk.lower()
    # very light check: token overlap of key words (split on punctuation/space)
    tokens = [t for t in ref.replace(";", " ").replace(",", " ").split() if len(t) > 3]
    return 1 if any(t in chunk_l for t in tokens) else 0

# ----------- A) Out-of-KB: % UNANSWERED -----------
def evaluate_out_of_kb():
    """
    Out-of-KB evaluation .

    Rules:
      - Strict refusal detection: ONLY exact 'UNANSWERED' is a refusal.
      - Keep refusals in the pool.
      - Scoring references:
          * Refusal      → pred='unanswered' vs ref="i'm sorry i don't have an answer."
          * Non-refusal  → pred=<answer>     vs ref="answer not expected in this set."
        This avoids both inflation (~0.85 when comparing refusal↔refusal) and collapse to 0.0
        (answer↔empty), giving stable low-but-nonzero scores.

    Outputs:
      - OUT_OF_KB_RESULTS (summary CSV)
      - results_out_of_kb_detailed.csv (per-question)
      - results_out_of_kb_debug.txt (phase/debug log)
    """


    # ---------- paths ----------
    summary_path = OUT_OF_KB_RESULTS
    details_path = os.path.join(os.path.dirname(OUT_OF_KB_RESULTS), "results_out_of_kb_detailed.csv")
    debug_path   = os.path.join(os.path.dirname(OUT_OF_KB_RESULTS), "results_out_of_kb_debug.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # ---------- helpers ----------
    REFUSAL_REF = "i'm sorry i don't have an answer."
    NONREF_REF  = "answer not expected in this set."

    def _clean(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "").strip().lower())

    def _is_strict_unanswered(t: str) -> bool:
        return (_clean(t) == "unanswered")  # exact token only

    # ---------- metrics ----------
    bert_metric  = load_metric("bertscore")
    rouge_metric = load_metric("rouge")

    def _compute(preds, refs):
        if not preds:
            return 0.0, 0.0
        b = bert_metric.compute(predictions=preds, references=refs, lang="en")
        r = rouge_metric.compute(predictions=preds, references=refs,
                                 rouge_types=["rouge1"], use_stemmer=True)
        bert = float(np.mean(b["f1"]))
        rouge = float(r["rouge1"].mid.fmeasure) if hasattr(r["rouge1"], "mid") else float(r["rouge1"])
        return round(bert, 4), round(rouge, 4)

    # ---------- setup ----------
    retriever, llm = setup_rag()
    items = load_jsonl(OUT_OF_KB_PATH)

    rows_summary, rows_detail, dbg = [], [], []

    for k in TOPK_LIST:
        total = unanswered = 0
        preds_eval, refs_eval = [], []

        dbg.append(f"\n===== DEBUG k={k} =====")

        for obj in tqdm(items, desc=f"Out-of-KB k={k}"):
            q   = obj["question"]
            ref = obj.get("reference_answer", "").strip() or REFUSAL_REF

            # Retrieve
            try:
                docs = retriever.get_relevant_documents(q, k=k)
            except TypeError:
                if hasattr(retriever, "search_kwargs"):
                    retriever.search_kwargs["k"] = k
                docs = retriever.get_relevant_documents(q)
            chunks = [getattr(d, "page_content", str(d)) for d in (docs or [])]

            # Predict
            if not chunks:
                pred_raw, reason = "UNANSWERED", "no_chunks"
            else:
                pred_raw = llm_answer(llm, q, chunks)
                reason = "model"

            # Classify
            is_refusal = _is_strict_unanswered(pred_raw)
            if is_refusal:
                unanswered += 1
                pred_used = "unanswered"
                ref_used  = REFUSAL_REF
            else:
                pred_used = _clean(pred_raw)
                ref_used  = NONREF_REF

            preds_eval.append(_clean(pred_used))
            refs_eval.append(_clean(ref_used))

            rows_detail.append({
                "set"     : "out_of_kb",
                "k"       : k,
                "question": q,
                "answer"  : ref,
                "pred"    : pred_raw,
                "reason"  : reason if not is_refusal else "refusal"
            })

            dbg.append(
                f"Q: {q[:72]}… | chunks={len(chunks)} | refusal={is_refusal} "
                f"| pred_used='{pred_used[:60]}' | ref_used='{ref_used}'"
            )

            total += 1

        pct_unans = 100.0 * unanswered / max(1, total)
        bert_f1, rouge1 = _compute(preds_eval, refs_eval)

        rows_summary.append({
            "k": k,
            "total": total,
            "unanswered": unanswered,
            "percent_unanswered": f"{pct_unans:.2f}",
            "BERTScore_F1_avg": f"{bert_f1:.4f}",
            "ROUGE1": f"{rouge1:.4f}"
        })

        dbg.append(f"Total={total}, Unanswered={unanswered} ({pct_unans:.2f}%)")
        dbg.append(f"BERTScore-F1={bert_f1:.4f}  ROUGE-1={rouge1:.4f}")

    # ---------- write outputs ----------
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["k", "total", "unanswered", "percent_unanswered",
                           "BERTScore_F1_avg", "ROUGE1"]
        )
        writer.writeheader()
        writer.writerows(rows_summary)

    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["set", "k", "question", "answer", "pred", "reason"]
        )
        writer.writeheader()
        writer.writerows(rows_detail)

    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dbg))

    print(f"\nOut-of-KB evaluation complete: {summary_path}")
    print(f"   Details: {details_path}")
    print(f"   Debug: {debug_path}")
    return rows_detail

# ----------- B) Inferred: NDCG, BERTScore, ROUGE-1 -----------
def evaluate_inferred():
    retriever, llm = setup_rag()
    items = load_jsonl(INFERRED_PATH)

    bert = load_metric("bertscore")
    rouge = load_metric("rouge")

    rows = []
    detail_rows = []

    for k in TOPK_LIST:
        preds, refs = [], []
        ndcgs = []

        for obj in tqdm(items, desc=f"Inferred k={k}"):
            q  = obj["question"]
            ref = obj.get("reference_answer","").strip()

            try:
                docs = retriever.get_relevant_documents(q, k=k)
            except TypeError:
                retriever.search_kwargs["k"] = k
                docs = retriever.get_relevant_documents(q)

            chunks = [d.page_content for d in docs] if docs else []
            pred = llm_answer(llm, q, chunks)

            # NDCG proxy: binary relevance if a chunk contains tokens from ref
            rels = [relevance_binary(ref, c) for c in chunks]
            ndcgs.append(ndcg_at_k(rels[:k]))

            preds.append(pred)
            refs.append(ref)

            detail_rows.append({"set":"inferred","k":k,"question":q,"answer":ref,"pred":pred,"rels":sum(rels)})

        # Metrics
        bert_res = bert.compute(predictions=preds, references=refs, lang="en")
        rouge_res = rouge.compute(predictions=preds, references=refs)
        # ROUGE-1 only
        rouge1 = rouge_res.get("rouge1", 0.0)

        rows.append({
            "k": k,
            "NDCG@k_avg": f"{(sum(ndcgs)/max(1,len(ndcgs))):.4f}",
            "BERTScore_F1_avg": f"{(sum(bert_res['f1'])/len(bert_res['f1'])):.4f}",
            "ROUGE1": f"{rouge1:.4f}",
            "num_questions": len(items)
        })

    os.makedirs(os.path.dirname(INFERRED_RESULTS), exist_ok=True)
    with open(INFERRED_RESULTS,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k","NDCG@k_avg","BERTScore_F1_avg","ROUGE1","num_questions"])
        w.writeheader(); w.writerows(rows)

    return detail_rows

# ----------- C) Known: NDCG, BERTScore, ROUGE-1 -----------
def evaluate_known():
    retriever, llm = setup_rag()
    items = load_jsonl(KNOWN_PATH)

    bert = load_metric("bertscore")
    rouge = load_metric("rouge")

    rows = []
    detail_rows = []

    for k in TOPK_LIST:
        preds, refs = [], []
        ndcgs = []

        for obj in tqdm(items, desc=f"Known k={k}"):
            q   = obj["question"]
            ref = obj.get("reference_answer", "").strip()

            try:
                docs = retriever.get_relevant_documents(q, k=k)
            except TypeError:
                retriever.search_kwargs["k"] = k
                docs = retriever.get_relevant_documents(q)

            chunks = [d.page_content for d in docs] if docs else []
            pred = llm_answer(llm, q, chunks)

            rels = [relevance_binary(ref, c) for c in chunks]
            ndcgs.append(ndcg_at_k(rels[:k]))

            preds.append(pred)
            refs.append(ref)

            detail_rows.append({
                "set":"known","k":k,"question":q,"answer":ref,"pred":pred,"rels":sum(rels)
            })

        bert_res  = bert.compute(predictions=preds, references=refs, lang="en")
        rouge_res = rouge.compute(predictions=preds, references=refs)
        rouge1    = rouge_res.get("rouge1", 0.0)

        rows.append({
            "k":k,
            "NDCG@k_avg":f"{(sum(ndcgs)/max(1,len(ndcgs))):.4f}",
            "BERTScore_F1_avg":f"{(sum(bert_res['f1'])/len(bert_res['f1'])):.4f}",
            "ROUGE1":f"{rouge1:.4f}",
            "num_questions":len(items)
        })

    os.makedirs(os.path.dirname(KNOWN_RESULTS), exist_ok=True)
    with open(KNOWN_RESULTS,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k","NDCG@k_avg","BERTScore_F1_avg","ROUGE1","num_questions"])
        w.writeheader(); w.writerows(rows)

    return detail_rows

if __name__ == "__main__":
    details = []
    details += evaluate_out_of_kb()
    details += evaluate_inferred()
    details += evaluate_known()

    # Write per-question log for debugging / appendix
    with open(DETAILS_LOG,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set","k","question","answer","pred","reason","rels"])
        w.writeheader()
        for r in details:
            if "reason" not in r: r["reason"] = ""
            if "rels" not in r: r["rels"] = ""
            w.writerow(r)

    print(f"\n✅ Saved:\n- {OUT_OF_KB_RESULTS}\n- {INFERRED_RESULTS}\n- {KNOWN_RESULTS}\n- {DETAILS_LOG}")
