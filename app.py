import os
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="HR Chatbot", layout="centered")
st.title("💼 HR Chatbot")
st.markdown("Hey There! I am your HR assistant! If you have any queries regarding HR policies, feel free to ask me!")

load_dotenv(".env")
api_key = os.getenv("OPENROUTER_API_KEY")

pdf_folder_path = "rag_pdf"
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


def evaluate_salary_query(query):
    try:
        match = re.search(r'entry|mid|senior', query.lower())
        salary_match = re.search(r'₹?(\d+(?:\.\d+)?)\s*l', query.lower())
        if not (match and salary_match):
            return None
        role = match.group(0)
        salary_lpa = float(salary_match.group(1))
        gross = salary_lpa * 100000
        if role == "entry":
            basic = 0.4 * gross
            hra = 0.4 * basic
        elif role == "mid":
            basic = 0.35 * gross
            hra = 0.4 * basic
        else:
            basic = 0.3 * gross
            hra = 0.5 * basic
        pf = 0.12 * basic
        gratuity = 0.0481 * basic
        special = gross - (basic + hra + pf + gratuity)
        return (
            f"For {role.capitalize()} role, approximate salary breakdown:\n"
            f"- Basic: ₹{round(basic)}\n"
            f"- HRA: ₹{round(hra)}\n"
            f"- PF: ₹{round(pf)}\n"
            f"- Gratuity: ₹{round(gratuity)}\n"
            f"- Special Allowance: ₹{round(special)}"
        )
    except:
        return None


def evaluate_tax_query(query):
    try:
        salary_match = re.search(r'₹?(\d+(?:\.\d+)?)\s*l', query.lower())
        if not salary_match:
            return None
        salary = float(salary_match.group(1)) * 100000
        tax = 0
        if salary <= 250000:
            tax = 0
        elif salary <= 500000:
            tax = 0.05 * (salary - 250000)
        elif salary <= 1000000:
            tax = (0.05 * 250000) + (0.10 * (salary - 500000))
        elif salary <= 1500000:
            tax = (0.05 * 250000) + (0.10 * 500000) + (0.15 * (salary - 1000000))
        else:
            tax = (0.05 * 250000) + (0.10 * 500000) + (0.15 * 500000) + (0.30 * (salary - 1500000))
        cess = 0.04 * tax
        total = tax + cess
        return f"Income Tax: ₹{round(tax)}, Cess: ₹{round(cess)}, Total Tax Payable: ₹{round(total)}"
    except:
        return None


def evaluate_bonus_query(query):
    try:
        if "festival bonus" in query.lower():
            basic_match = re.search(r'₹?(\d+(?:\.\d+)?)\s*l', query.lower())
            if not basic_match:
                return "Festival Bonus: Fixed ₹5,000 (or 5% of one month's basic, whichever is lower)."
            basic = float(basic_match.group(1)) * 100000 / 12
            five_percent = 0.05 * basic
            bonus = min(5000, five_percent)
            return f"Festival Bonus = ₹{round(bonus)}"
        if "performance bonus" in query.lower():
            return "Annual Performance Bonus: Paid in April-May, eligibility requires minimum 6 months service."
        return None
    except:
        return None


with st.spinner("Loading HR documents and building memory..."):
    all_texts = load_pdfs(pdf_folder_path)
    chunks = semantic_chunking(all_texts)
    index, _ = build_faiss_index(chunks)

query = st.text_input("Enter your HR-related question:")

if query:
    with st.spinner("Thinking..."):
        tax = evaluate_tax_query(query)
        salary = evaluate_salary_query(query)
        bonus = evaluate_bonus_query(query)
        if tax:
            st.markdown("**Answer:**")
            st.write(tax)
        elif salary:
            st.markdown("**Answer:**")
            st.write(salary)
        elif bonus:
            st.markdown("**Answer:**")
            st.write(bonus)
        else:
            context_chunks = retrieve_chunks(query, chunks, index)
            prompt = build_prompt(query, context_chunks)
            answer = query_openrouter(prompt)
            st.markdown("**Answer:**")
            st.write(answer.strip())
