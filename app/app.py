from flask import Flask, render_template, request, redirect, session, url_for
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session-based chat history


# 1: Load retriever and LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

retriever = FAISS.load_local(
    "retriever/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
).as_retriever(search_kwargs={"k": 3})  # retrieve top 3 relevant chunks

llm = Ollama(model="mistral")


# 2: Define a clean retrieval-aware prompt
prompt = ChatPromptTemplate.from_template("""
You are SARAG, a helpful study assistant that uses the retrieved context to answer student questions clearly and concisely.
If the answer cannot be found in the context, say you don’t know — do not make up information.

Previous conversation:
{history}

Context:
{context}

Question:
{input}
""")


# 3: Build the RAG chain 
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


# 4: Routes 
@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize chat history
    if "history" not in session:
        session["history"] = []

    chat_history = session["history"]

    if request.method == "POST":
        query = request.form["query"]

        # Build readable conversation context for LLM
        conversation_context = "\n".join(
            [f"User: {msg['user']}\nSARAG: {msg['bot']}" for msg in chat_history]
        )

        try:
            # Get RAG response
            response = rag_chain.invoke({
                "input": query,
                "history": conversation_context
            })

            # Extract answer text
            answer = response.get("answer", "Sorry, I couldn’t find an answer.")

            # Gather source info from retrieved docs
            sources = []
            if "context" in response and response["context"]:
                for doc in response["context"]:
                    metadata = getattr(doc, "metadata", {})
                    src = metadata.get("source") or metadata.get("file") or "Unknown"
                    sources.append(src)
            sources = list(set(sources))  # remove duplicates

            # Append sources to final answer neatly
            if sources:
                answer += f"\n\n📚 Sources: {', '.join(sources)}"

        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"
            sources = []

        # Store turn in chat history
        chat_history.append({"user": query, "bot": answer})
        session["history"] = chat_history

        return render_template("chat.html", chat_history=chat_history)

    # GET: render chat history (or greeting)
    return render_template("chat.html", chat_history=chat_history)


@app.route("/clear", methods=["POST"])
def clear():
    session.pop("history", None)  # match actual session key
    return redirect(url_for("index"))




if __name__ == "__main__":
    print("🚀 SARAG is running on http://127.0.0.1:5000")
    app.run(debug=True)
