from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# 1. Load the saved Chroma vector database and local Ollama LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = OllamaLLM(model="llama3.2", temperature=0.1)

# 2. Ask a question ONLY answerable from your specific exam PDF
query = (
    "In the Collaborative Unmanned Ground and Aerial Systems Technical Screening Examination, "
    "who is the candidate listed on the exam paper, what is the exact Theory vs Practical marks split, "
    "and what are the instructions for Part A?"
)

# 3. Test Retrieval Step (Top-3 most relevant chunks)
print("==========================================")
print("STEP 1: TESTING VECTOR RETRIEVAL (TOP-3)")
print("==========================================")
retrieved_docs = vector_db.similarity_search(query, k=3)
for i, doc in enumerate(retrieved_docs):
    print(f"\n[Retrieved Chunk {i+1} | Page: {doc.metadata.get('page', 'N/A')}]")
    print(doc.page_content[:300].strip() + "...")

# 4. Run Plain LLM (WITHOUT RAG)
print("\n==========================================")
print("STEP 2: WITHOUT RAG (PLAIN LLM RESPONSE)")
print("==========================================")
plain_response = llm.invoke(query)
print(plain_response)

# 5. Run Grounded LLM (WITH RAG)
context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
rag_prompt = f"""You are a helpful AI assistant. Answer the user's question strictly using ONLY the provided context below.
If the answer is not present in the context, state "I don't know based on the provided documents."

Context:
{context}

Question: {query}
Answer:"""

print("\n==========================================")
print("STEP 3: WITH RAG (GROUNDED RESPONSE)")
print("==========================================")
rag_response = llm.invoke(rag_prompt)
print(rag_response)
