import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load PDFs from the data folder
print("Loading PDFs from data/ ...")
loader = PyPDFDirectoryLoader("data/")
docs = loader.load()
print(f"Loaded {len(docs)} pages.")

# 2. Chunk sensibly (~300-800 tokens with overlap)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # ~250-300 tokens
    chunk_overlap=200,    # Preserves context across boundaries
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")

# 3. Inspect two sample chunks manually (Task Requirement)
if len(chunks) >= 2:
    print("\n--- SAMPLE CHUNK 1 ---")
    print(chunks[0].page_content[:400])
    print("\n--- SAMPLE CHUNK 2 ---")
    print(chunks[1].page_content[:400])

# 4. Generate embeddings and store in local ChromaDB
print("\nGenerating embeddings and saving to ./chroma_db ...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Success! Vector database saved to ./chroma_db.")
