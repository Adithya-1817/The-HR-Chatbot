import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Page configuration
st.set_page_config(page_title="HR Chatbot", layout="centered")
st.title("💼 HR Chatbot")
st.markdown("Hey There! I am your HR assistant! If you have any queries regarding the HR policies, feel free to ask me!")

# Load environment variables
load_dotenv(".env")
api_key = os.getenv("OPENROUTER_API_KEY")

# Define folder containing the PDF
pdf_folder_path = "rag_pdf"

# Global model instance (DO NOT pass it to cached functions)
model = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_pdfs(folder_path):
    all_text = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            all_text.append(text)
    return all_text

@st.cache_resource
def semantic_chunking(texts, threshold=0.7):
    paragraphs = []
    for text in texts:
        for para in text.split("\n\n"):
            if para.strip():
                paragraphs.append(para.strip())

    embeddings = model.encode(paragraphs)
    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i])
        if sim > threshold:
            current_chunk.append(paragraphs[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [paragraphs[i]]
    chunks.append(" ".join(current_chunk))

    return chunks

@st.cache_resource
def build_faiss_index(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query, chunks, index, top_k=2):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def build_prompt(query, context_chunks, max_chars=6000):
    context = "\n\n".join(context_chunks)
    if len(context) > max_chars:
        context = context[:max_chars]
    return f"""You are a helpful HR assistant. Use only the context below to answer the user's question. Do not guess or add anything beyond the provided context. Do not say "which in the provided context is". If the answer is not found in the context, say 'Sorry! The provided content doesn’t have the information you are looking for.'

Context:
{context}

Question: {query}

Answer:"""

def query_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    res_json = response.json()
    if response.status_code == 200 and 'choices' in res_json:
        return res_json['choices'][0]['message']['content']
    return "⚠️ Error: " + res_json.get('error', {}).get('message', 'Unknown error.')

# Load PDFs and index memory
with st.spinner("Loading HR documents and building memory..."):
    all_texts = load_pdfs(pdf_folder_path)
    chunks = semantic_chunking(all_texts)
    index, _ = build_faiss_index(chunks)

# User query input
query = st.text_input("Enter your HR-related question:")

if query:
    with st.spinner("Thinking..."):
        context_chunks = retrieve_chunks(query, chunks, index)
        prompt = build_prompt(query, context_chunks)
        answer = query_openrouter(prompt)
        st.markdown("**Answer:**")
        st.write(answer.strip())
