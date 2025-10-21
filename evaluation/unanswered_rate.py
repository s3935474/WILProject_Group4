import csv, os, time
from datasets import load_dataset
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# ---------- Config ----------
INDEX_PATH = "retriever/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
TOPK_LIST = [1, 3, 5]            # run like Walert for different k
MAX_QUESTIONS = 150              # limit for speed (increase if you want)
OUTPUT_CSV = "evaluation/results_unanswered.csv"
SEED = 42

# Force the model to say UNANSWERED if context is insufficient
SYSTEM_INSTRUCTION = (
    "You are a strict QA assistant. Use ONLY the provided context. "
    "If the context is insufficient to answer, reply with exactly the single word: UNANSWERED."
)

PROMPT_TEMPLATE = """{system}

Context:
{context}

Question: {question}

Answer:"""

# ---------- Helpers ----------
def setup_rag():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    retriever = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    ).as_retriever()
    llm = Ollama(model=OLLAMA_MODEL)
    return retriever, llm

def run_one(llm, question, context_chunks):
    # Join top-k chunks with separators
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
    prompt = PROMPT_TEMPLATE.format(system=SYSTEM_INSTRUCTION, context=context, question=question)
    try:
        # LangChain Ollama LLM acts like a Callable[str]
        output = llm(prompt)
        if hasattr(output, "content"):   # newer LC message object
            text = (output.content or "").strip()
        else:                             # plain string
            text = (output or "").strip()
        return text
    except Exception as e:
        return f"__ERROR__: {e}"

def unanswered_pred(text):
    if not text or text.strip() == "":
        return True
    normalized = text.strip().lower()
    return normalized == "unanswered"

def evaluate_split(split_name="test"):
    print(f"Loading dataset split: {split_name}")
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    data = ds[split_name]

    # Trim for speed if wanted
    n = min(len(data), MAX_QUESTIONS)
    questions = [data[i]["question"] for i in range(n)]

    retriever, llm = setup_rag()

    rows = []
    for k in TOPK_LIST:
        print(f"\nEvaluating % unanswered for k={k} on {n} questions...")
        unanswered = 0
        total = 0

        # set retriever k each loop (LangChain retriever supports variable k via get_relevant_documents)
        for q in tqdm(questions, total=n):
            # retrieve
            try:
                docs = retriever.get_relevant_documents(q, k=k)
            except TypeError:
                # Some LC versions use search_kwargs
                retriever.search_kwargs["k"] = k
                docs = retriever.get_relevant_documents(q)

            chunks = [d.page_content for d in docs] if docs else []

            if not chunks:
                unanswered += 1
                total += 1
                continue

            # generate
            out = run_one(llm, q, chunks)

            # mark unanswered
            if out.startswith("__ERROR__"):
                # count errors as unanswered to be conservative
                unanswered += 1
            elif unanswered_pred(out):
                unanswered += 1

            total += 1

        rate = 100.0 * unanswered / max(1, total)
        print(f"k={k}: unanswered {unanswered}/{total} = {rate:.2f}%")
        rows.append({"k": k, "total": total, "unanswered": unanswered, "percent_unanswered": f"{rate:.2f}"})

    # write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "total", "unanswered", "percent_unanswered"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    evaluate_split("test")
