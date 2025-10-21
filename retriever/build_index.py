from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 1: Load dataset
dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

# 2: Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 3: Create documents *with true Hugging Face IDs*
docs = []
for i, data in enumerate(dataset["test"]):
    question = data["question"]
    answer = data["answer"]
    entry_id = data.get("id", i)   # use Hugging Face’s actual id column

    docs.append(Document(
        page_content=f"Question: {question}\nAnswer: {answer}",
        metadata={
            "source": f"https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia/tree/main#entry-{entry_id}",
            "title": question[:60] + "..."
        }
    ))

# 4: Split into retrievable chunks
chunked_docs = splitter.split_documents(docs)

# 5: Build FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunked_docs, embeddings)
vectorstore.save_local("retriever/faiss_index")

print("✅ Retriever index built and saved with true Hugging Face citations!")
