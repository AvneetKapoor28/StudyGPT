import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

def build_faiss():
    documents = []

    # ✅ Loop through all PDFs in the data folder
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, file_name)
            print(f"📄 Loading: {file_path}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        print("❌ No PDF files found in the data folder!")
        return

    print(f"✅ Loaded {len(documents)} documents. Splitting into chunks...")

    # ✅ Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks.")

    # ✅ Initialize local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},  # or "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True}
    )

    # ✅ Build FAISS index
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)
    print(f"✅ FAISS index built and saved to '{INDEX_DIR}'.")

if __name__ == "__main__":
    build_faiss()
