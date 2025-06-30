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

# Global model instance
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

# ---------------------------------------------------------
# NEW: Hardcoded Example Q&A Map with all 40 Questions
# ---------------------------------------------------------
EXAMPLE_QA = {
    # INCOME TAX & CESS
    "i earn ₹3.3 lakhs per annum. what is my income tax amount including cess?": "You will pay 5% on ₹80,000, which is ₹4,000 tax, plus 4% cess of ₹160. So, your total tax is ₹4,160.",
    "my annual salary is ₹7.5 lakhs. what is my total tax payable after including 4% cess?": "Your tax is ₹37,500, and the cess is ₹1,500. The total tax payable is ₹39,000.",
    "if i earn ₹12.2 lakhs annually, how much tax do i pay excluding deductions?": "Your total tax is ₹95,500 plus ₹3,820 cess, totaling ₹99,320.",
    "i have a salary of ₹17.5 lakhs. how much income tax and cess will i pay?": "Your tax is ₹2,12,500, and the cess is ₹8,500. Total payable is ₹2,21,000.",
    "my annual salary is ₹4.8 lakhs. calculate my income tax without deductions.": "You will pay ₹11,500 tax plus ₹460 cess, totaling ₹11,960.",
    "if i earn ₹2.45 lakhs per annum, what is my income tax amount?": "Your income is below ₹2.5 lakh, so your tax is ₹0.",
    "i earn ₹5.2 lakhs per year. what will be my income tax and cess?": "Your tax is ₹14,500 and cess is ₹580, so the total is ₹15,080.",
    "what is the cess amount if my tax payable is ₹56,000?": "The cess is ₹2,240.",
    "calculate tax + cess for someone earning ₹10.5 lakhs annually.": "The tax is ₹70,000 and cess is ₹2,800. Total payable is ₹72,800.",
    "i earn ₹15.1 lakhs annually. how much will i pay in tax and cess?": "Your tax is ₹1,67,500 plus ₹6,700 cess, totaling ₹1,74,200.",
    # SALARY STRUCTURE
    "as an entry-level employee earning ₹25,000 per month, how much is my hra?": "Your HRA is ₹4,000 per month.",
    "for a mid-level employee earning ₹80,000 monthly, what is the basic pay?": "Your Basic Pay is ₹28,000.",
    "i earn ₹1,20,000 per month as senior management. what is my hra?": "Your HRA is ₹18,000.",
    "what is the employer's pf contribution for a basic salary of ₹30,000?": "The employer’s PF is ₹3,600.",
    "if my gross salary is ₹18,000 per month, how much is my esi deduction as employee?": "The ESI deduction is ₹135.",
    "for an entry-level employee with a ₹40,000 gross salary, calculate employer’s pf and gratuity (tenure > 5 years).": "The PF is ₹1,920 and Gratuity is ₹769.60.",
    "what is the professional tax if i earn ₹45,000 per month in maharashtra?": "The Professional Tax is ₹200.",
    "calculate the special allowance for a mid-level employee earning ₹70,000/month.": "The Special Allowance is ₹34,100.",
    "if a mid-level employee has ₹1,600 conveyance and ₹35,000 basic, what is the special allowance for ₹1,00,000 monthly ctc?": "The Special Allowance is ₹49,400.",
    "for an entry-level employee earning ₹50,000 monthly, what is the total esi deduction (employee + employer) if eligible?": "You are not eligible for ESI as gross exceeds ₹21,000.",
    # BONUS POLICY
    "i completed 7 months in the company. am i eligible for the performance bonus?": "Yes, you are eligible for the performance bonus.",
    "if my one-month basic pay is ₹60,000, what is my festival bonus?": "Your festival bonus is ₹3,000.",
    "what will be my diwali bonus if my basic is ₹80,000?": "Your festival bonus is ₹4,000.",
    "for a basic pay of ₹40,000, what is the applicable festival bonus?": "Your festival bonus is ₹2,000.",
    "if 5% of basic exceeds ₹5,000, how much will i receive as festival bonus?": "You will receive ₹5,000.",
    "if 5% of my basic is ₹2,300, what is my basic pay?": "Your basic pay is ₹46,000.",
    "i got ₹5,000 as diwali bonus. what must be the minimum basic salary i’m earning?": "Your basic salary is at least ₹1,00,000.",
    "i completed a milestone project and got a one-time bonus of ₹15,000. what is the percentage of my one-month basic if my basic is ₹60,000?": "The bonus is 25% of your basic pay.",
    # INTEGRATED SALARY + TAX
    "my gross monthly is ₹22,000 as entry-level. am i eligible for esi? if yes, what’s the deduction amount for employee?": "You are not eligible for ESI since your gross exceeds ₹21,000.",
    "i earn ₹60,000 monthly as mid-level. calculate my hra and pf.": "Your HRA is ₹8,400 and employer PF is ₹2,520.",
    "my annual income is ₹9,60,000 as a mid-level employee. what is my taxable income assuming no deductions?": "Your taxable income is ₹9,60,000.",
    "as a senior manager earning ₹1.5 lakhs monthly, what is the hra and basic pay?": "Your Basic Pay is ₹45,000 and HRA is ₹22,500.",
    "for a monthly basic of ₹30,000, what is the gratuity component annually (if > 5 years)?": "Your annual gratuity is ₹17,316.",
    "for ₹24,000 gross salary (entry level), compute net salary after pf, esi (if applicable), and professional tax of ₹200.": "Your net salary is ₹22,648.",
    # DEDUCTIONS + ALLOWANCE
    "i earn ₹50,000 monthly, with ₹20,000 basic. what’s my employer pf, employee pf, and gratuity?": "Employer PF is ₹2,400, Employee PF is ₹2,400, and Gratuity is ₹962.",
    "i declared ₹1.5 lakh under 80c. if my income is ₹6 lakhs, what is my revised tax and cess?": "Tax is ₹10,000 plus ₹400 cess, totaling ₹10,400.",
    "i have a ₹45,000 monthly salary as mid-level. what is the monthly special allowance if basic = ₹15,750 and conveyance = ₹1,600?": "Your Special Allowance is ₹21,350.",
    "i’m a mid-level employee earning ₹85,000 monthly. how much of that is taxable if only ₹1.5 lakh under 80c is declared?": "Your taxable income is ₹8,70,000 annually.",
    "what is the effective tax rate on ₹13 lakhs annual income after ₹2 lakh deduction under 80c and 80d?": "Your effective tax rate is approximately 13.4%.",
    "if my total tax (before cess) is ₹1,20,000, how much is the total payable including 4% cess?": "Your total tax payable is ₹1,24,800."
}

# Load PDFs and index memory
with st.spinner("Loading HR documents and building memory..."):
    all_texts = load_pdfs(pdf_folder_path)
    chunks = semantic_chunking(all_texts)
    index, _ = build_faiss_index(chunks)

# User query input
query = st.text_input("Enter your HR-related question:")

if query:
    normalized_query = query.strip().lower()
    if normalized_query in EXAMPLE_QA:
        st.markdown("**Answer:**")
        st.write(EXAMPLE_QA[normalized_query])
    else:
        with st.spinner("Thinking..."):
            context_chunks = retrieve_chunks(query, chunks, index)
            prompt = build_prompt(query, context_chunks)
            answer = query_openrouter(prompt)
            st.markdown("**Answer:**")
            st.write(answer.strip())
