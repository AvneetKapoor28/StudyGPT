import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load .env file
load_dotenv()

# ✅ Make sure the API key exists
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found! Please add it to your .env file.")
    st.stop()

INDEX_DIR = "faiss_index"

st.title("📚 StudyGPT – JEE RAG Assistant")
st.write("Ask conceptual questions from NCERT / JEE material.")

# Load embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 20})

# Define prompt template
prompt = PromptTemplate(
    template="You are the best JEE trainer in the world. Answer the question asked by the student using the context provided. Answer in a way that the student's does not have any coceptual doubts remaining. Wherever possible quote the context provided in markdown format as a part of the flow of your explanation. Also, if the question of the student is irrelevant to JEE / academics respond with a message that convinces the student to get back to studying\nContext:\n{context}\n\nQuestion: {question}",
    input_variables=["context", "question"]
)

# Initialize Gemini LLM (API key is automatically picked up)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0
)

# Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# UI for query input
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
    st.write("### Answer")
    st.write(answer)
