from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

app = Flask(__name__)

# Load retriever and LLM 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

retriever = FAISS.load_local(
    "retriever/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
).as_retriever()

llm = Ollama(model="mistral")

prompt = ChatPromptTemplate.from_template("""
You are a helpful study assistant. Use the following retrieved context to answer the question.
If the answer is not in the provided context, say you don’t know.

Context:
{context}

Question:
{input}
""")

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        try:
            response = rag_chain.invoke({"input": query})
            answer = response.get("answer", "Sorry, I couldn’t find an answer.")
        except Exception as e:
            answer = f"Error: {str(e)}"
        return render_template("chat.html", query=query, answer=answer)
    return render_template("chat.html", query=None, answer=None)


if __name__ == "__main__":
    app.run(debug=True)
