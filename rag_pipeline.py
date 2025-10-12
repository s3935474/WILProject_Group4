from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# loading the FAISS retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local(
    "retriever/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
).as_retriever()

# connecting to local Ollama model
llm = Ollama(model="mistral")

# building retrieval augmented generation pipeline
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# testing a query
query = "What is the capital of France?"
result = qa.run(query)

print("Question:", query)
print("Answer:", result)
