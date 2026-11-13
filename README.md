# Local Retrieval-Augmented Generation (RAG) System

A fully local Retrieval-Augmented Generation (RAG) pipeline built with **Python**, **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Ollama (`llama3.2`)**. This system ingests private PDF documents, chunks and embeds them into a local vector store, retrieves top-\(k\) relevant passages, and grounds LLM answers strictly in the retrieved context.

---

## Architecture & Tech Stack
* **Document Loader:** `PyPDFDirectoryLoader` (22-page technical examination document)
* **Chunking Strategy:** `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200` across paragraph and sentence boundaries -> 47 chunks)
* **Embeddings:** `all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`
* **Vector Database:** Local persistent `ChromaDB` (`./chroma_db`)
* **LLM Inference:** Local `llama3.2` via `Ollama` (`temperature=0.1`)

---

## Setup & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is running locally with `llama3.2`:**
   ```bash
   ollama pull llama3.2
   ```

3. **Ingest documents & build the vector database:**
   Place your PDF files inside `data/` and run:
   ```bash
   python3 ingest.py
   ```

4. **Run retrieval & grounding evaluation:**
   ```bash
   python3 evaluate.py
   ```

---

## Proof of Grounding: With RAG vs. Without RAG

To verify that answers are grounded in the ingested document rather than parametric training data, the system was tested on a query whose answer exists **only** inside the private 22-page PDF (`ugv_uav_exam.pdf`).

### Test Query
> *"In the Collaborative Unmanned Ground and Aerial Systems Technical Screening Examination, who is the candidate listed on the exam paper, what is the exact Theory vs Practical marks split, and what are the instructions for Part A?"*

<img width="1206" height="913" alt="Screenshot 2026-11-13 161804" src="https://github.com/user-attachments/assets/1baed7a4-b284-4405-88f5-23872a3ed4a2" />


### Top Retrieved Context (`Page 0`)
```text
COLLABORATIVE UNMANNED GROUND AND AERIAL SYSTEMS
TECHNICAL SCREENING EXAMINATION
Name - Raj Thakur   Contact- +91 7906352166   Email - build.rajthakur@gmail.com

Duration 3 Hours Maximum Marks 100
Difficulty Moderate Theory / Practical 40 / 60
Instructions
- Attempt all questions in Part A.
- Document assumptions, test results and safety measures.
- Practical tasks are designed around benign applications such as inspection, mapping, search-and-rescue training, and environmental monitoring.
```

### Side-by-Side Comparison

| Mode | Model Output | Result |
| :--- | :--- | :--- |
| **Without RAG** *(Plain LLM)* | *"I can't provide information about a specific exam. Is there something else I can help you with?"* | **Failed** (No access to private document context) |
| **With RAG** *(ChromaDB + Context)* | *"The candidate listed on the exam paper is Raj Thakur. The Theory vs Practical marks split is 40/60. The instructions for Part A are: <br>• Attempt all questions in Part A.<br>• Document assumptions, test results, and safety measures.<br>• Practical tasks are designed around benign applications such as inspection, mapping, search-and-rescue training, and environmental monitoring."* | **Passed** (Accurately retrieved and grounded in Page 0) |
