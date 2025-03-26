import os
import json
import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up API keys using Streamlit secrets
GOOGLE_API_KEY = st.secrets["secrets"]["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["secrets"]["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["secrets"]["PINECONE_ENV"]

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index("matscibert-rag-norm")

# Load MatSciBERT model and tokenizer
model_name = "m3rg-iitd/matscibert"
tokenizer_mat = AutoTokenizer.from_pretrained(model_name)
model_mat = AutoModel.from_pretrained(model_name).to("cpu")

# Function to embed a query using MatSciBERT
def embed_query(query):
    inputs = tokenizer_mat(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cpu")
    with torch.no_grad():
        embeddings = model_mat(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.squeeze().cpu().numpy()

# Function to retrieve relevant documents from Pinecone
def retrieve_relevant_docs(query, index, top_k=3):
    query_vector = embed_query(query)
    response = index.query(vector=query_vector.tolist(), top_k=top_k, include_metadata=True)
    documents = [match['metadata']['text'] for match in response['matches']]
    return documents if documents else None

# Function to check hallucination using cosine similarity
def check_hallucination(response, context_list):
    response_embedding = embed_query(response)
    max_similarity = max(
        cosine_similarity([response_embedding], [embed_query(context)])[0][0] for context in context_list
    )
    return max_similarity >= 0.6  # Threshold for hallucination detection

# Function to generate response using Gemini 2.0 Flash
def generate_response_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# Implementing RAG + Self-Reflection Pipeline with Gemini
def rag_reflection(query, index, top_k=3):
    context = retrieve_relevant_docs(query, index, top_k=top_k)
    if not context:
        return "⚠️ No relevant documents found. Try refining your query."

    input_text = (
        "You are an expert in battery materials. Answer the following query accurately:\n\n"
        f"Query: {query}\n\n"
        "Relevant Context:\n" + "\n".join(context) + "\n\nProvide a concise and factual answer:\n"
    )
    response = generate_response_with_gemini(input_text)
    return response if check_hallucination(response, context) else "⚠️ The response might contain hallucinations. Please verify the information."

# Streamlit UI
st.title("🔍 RAG + Self-Reflection Pipeline with Gemini")

# Initialize session state for storing queries and responses
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

query = st.text_input("Enter your query:", placeholder="What are the latest developments in lithium-ion battery materials?")

if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        response = rag_reflection(query, index)
        st.session_state.qa_history.append({"query": query, "response": response})
    st.write("### Response:")
    st.write(response)

# Show previously asked questions
if st.session_state.qa_history:
    st.subheader("📜 Previous Queries in This Session")
    for qa in st.session_state.qa_history:
        with st.expander(qa["query"]):
            st.write(qa["response"])

# Option to save history to a JSON file
if st.button("💾 Save Q&A to JSON"):
    file_path = "qa_history.json"
    with open(file_path, "w") as f:
        json.dump(st.session_state.qa_history, f, indent=4)
    st.success(f"Saved to {file_path}")

# Option to download the JSON file
if os.path.exists("qa_history.json"):
    with open("qa_history.json", "r") as f:
        st.download_button("📥 Download Q&A History", f, file_name="qa_history.json", mime="application/json")
