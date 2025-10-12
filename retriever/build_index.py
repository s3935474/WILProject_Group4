from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# loading the dataset
dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

# combining the question and answer into a single text block
texts = [f"Q: {d['question']} A: {d['answer']}" for d in dataset["test"]]

# spliting the text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents(texts)

# creating the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# building the FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("retriever/faiss_index")

print("Retriever index built and saved successfully!")
