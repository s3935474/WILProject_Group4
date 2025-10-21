# WIL Project Group 4 – Study Assistant RAG (SARAG)
Authors: Mikael Ali, Diamond, Jimmy 

Case Studies in Data Science – RMIT University  
Semester 2, 2025

---

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system called **SARAG (Study Assistant RAG)**.  
SARAG combines 
**LangChain** for pipeline orchestration 
**FAISS** for vector-based document retrieval and 
**HuggingFace sentence-transformer embeddings** for semantic similarity **Ollama’s Mistral model** for local LLM inference

The app includes a lightweight **Flask web interface** that allows users to ask questions and receive context-aware answers based on retrieved content.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
git clone https://github.com/s3935474/WILProject_Group4.git
cd WILProject_Group4

### 2️⃣ Create a virtual environment (venv)
python -m venv rag_env (name rag_env whatever you want)

### 3️⃣ Activate the environment
On Windows: .\rag_env\Scripts\activate
On macOS/Linux: source rag_env/bin/activate

### 4️⃣ Install dependencies 
pip install -r updated_requirements.txt 

### 5️⃣ Run the webapp with:
python app/app.py (assuming you are in the root directory)

### 6️⃣ Open browser and visit lnk provided:
default is http://127.0.0.1:5000

### 🗝️ Dependencies (found in requirements.txt)
1. flask
2. langchain-community
3. langchain-experimental
4. transformers
5. faiss-cpu 
6. datasets

### Additional Setup: Ollama (Local LLM Backend)

This project uses **Ollama** to run local LLMs such as Mistral or Llama 3.

#### 1️⃣ Install Ollama
Download and install Ollama from the official site:
👉 [https://ollama.ai/download](https://ollama.ai/download)

#### 2️⃣ Verify installation and run Ollama server
After installation, open a new terminal to run and confirm ollama:
ollama --version 

Then, ollama serve

Finally, ollama pull mistral (or any desired model)

To check if ollama is actively serving, search in the browser:
http://localhost:11434/api/tags
