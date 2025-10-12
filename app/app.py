from flask import Flask, request, render_template_string
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Load retriever and LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local(
    "retriever/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
).as_retriever()
llm = Ollama(model="mistral")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

HTML = """
<!DOCTYPE html>
<html>
<body style="font-family:sans-serif;">
<h2>Study Assistant RAG (SARAG)</h2>
<form method="POST">
  <input name="query" style="width:400px;" placeholder="Ask a question">
  <input type="submit" value="Ask">
</form>
{% if answer %}
  <h3>Answer:</h3>
  <p>{{ answer }}</p>
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    if request.method == "POST":
        query = request.form["query"]
        answer = qa.run(query)
    return render_template_string(HTML, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
