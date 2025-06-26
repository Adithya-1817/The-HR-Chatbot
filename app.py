import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Streamlit UI
st.set_page_config(page_title="HR Policy Chatbot", layout="centered")
st.title("💼 HR Policy Chatbot")
st.markdown("Hey There! I am your HR assistant! If you have any queries regarding the HR policies, feel free to ask me!")

# Load environment variable
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# PDF folder path
pdf_folder_path = "rag_pdf"

# Load PDFs and extract text
@st.cache_resource
def load_pdfs(folder_path):
    texts = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pdf"):
            with fitz.open(os.path.join(folder_path, file)) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                texts.append(text)
    return texts

# Split text into semantic chunks
@st.cache_resource
def chunk_texts(texts, chunk_size=500, overlap=50):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks

# Build FAISS index
@st.cache_resource
def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

# Retrieve top chunks
def get_top_chunks(query, chunks, index, model, top_k=2):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]

# Build prompt
def build_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    return f"""
You are a helpful HR assistant. Use only the context below to answer the question. Do not make up answers.
If the information is not found, reply with:
"Sorry! The provided content doesn’t have the information you are looking for."

Context:
{context}

Question: {query}

Answer:"""

# Query OpenRouter API
def query_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# Load everything
with st.spinner("Loading knowledge base..."):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = load_pdfs(pdf_folder_path)
    chunks = chunk_texts(texts)
    index, _ = build_faiss_index(chunks, model)

# User input
query = st.text_input("Ask your HR policy question:")

if query:
    with st.spinner("Thinking..."):
        top_chunks = get_top_chunks(query, chunks, index, model)
        prompt = build_prompt(top_chunks, query)
        answer = query_openrouter(prompt)
        st.markdown("**Answer:**")
        st.write(answer)
