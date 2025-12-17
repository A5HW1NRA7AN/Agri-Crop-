import os
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

# Configuration
DATA_PATH = "./data"
DB_PATH = "./vector_db"

def create_knowledge_base():
    # 1. Load PDFs
    print(f"Loading PDFs from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH}. Please put your agricultural PDFs here and rerun.")
        return

    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found. Add Agrios/Extension guides to the 'data' folder.")
        return

    print(f"Loaded {len(documents)} pages.")

    # 2. Chunk Data (Break text into learnable pieces)
    # 1000 chars is roughly one paragraph of medical/agri text.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Create Vector Store (The Brain)
    # We use FastEmbed (runs on CPU/GPU, very fast, lightweight)
    print("Creating Vector Database...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Persist to disk
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print(f"Success! Knowledge Base saved to {DB_PATH}")

if __name__ == "__main__":
    create_knowledge_base()