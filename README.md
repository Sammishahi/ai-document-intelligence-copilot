# AI Document Intelligence Copilot
### Multi-Document RAG & Evidence-Grounded Question Answering Platform

An end-to-end **Retrieval-Augmented Generation (RAG)** application for asking natural-language questions over multiple PDF documents.

The system does not directly ask an LLM to answer from its general knowledge. It first **retrieves relevant evidence from the uploaded documents**, re-ranks that evidence, and then generates an answer grounded in the retrieved context with source/page citations.

---

## 1. What Problem Does This Project Solve?

Large Language Models can generate answers, but they do not automatically know the contents of a user's private documents.

This project solves that problem by connecting an LLM to a user's PDF knowledge base.

For example, a user can upload:

- DenseNet paper
- Pixel-Adaptive Field-of-View paper
- ProsGradNet paper

and ask questions such as:

> What is the purpose of the CGSCR block in ProsGradNet?

The system finds the relevant part of the correct document and uses that evidence to generate the answer.

### In simple words

**User Question → Find relevant information → Re-rank evidence → Give evidence to LLM → Generate grounded answer + citation**

---

# 2. Complete Project Flow

```text
                         ┌──────────────────────┐
                         │        USER          │
                         │ Upload PDFs + Query  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Streamlit UI     │
                         │       Frontend       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │       Backend       │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
           ┌─────────────────┐              ┌─────────────────┐
           │  PDF INGESTION  │              │  USER QUESTION  │
           └────────┬────────┘              └────────┬────────┘
                    │                                │
                    ▼                                │
           ┌─────────────────┐                       │
           │ Text Extraction │                       │
           │   + OCR Fallback│                       │
           └────────┬────────┘                       │
                    │                                │
                    ▼                                │
           ┌─────────────────┐                       │
           │ Cleaning +      │                       │
           │ Chunking        │                       │
           └────────┬────────┘                       │
                    │                                │
                    ▼                                │
           ┌─────────────────┐                       │
           │   Embeddings    │                       │
           └────────┬────────┘                       │
                    │                                │
                    ▼                                │
           ┌─────────────────┐                       │
           │    ChromaDB     │◄──────────────────────┘
           │   Vector Store  │       Semantic Search
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Candidate       │
           │ Retrieval       │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Cross-Encoder   │
           │   Re-Ranking    │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Evidence        │
           │ Selection       │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │    Groq LLM     │
           │ Answer Generate │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────────────────┐
           │ Grounded Answer + Evidence │
           │ + Source / Page Citations  │
           └─────────────────────────────┘
```

---

# 3. How the System Works

## Step 1 — PDF Upload

The user uploads one or more PDF documents through Streamlit.

Example:

```text
DenseNet 121.pdf
Pixel-Adaptive_Field-of-View.pdf
ProsGradNet.pdf
```

Streamlit sends the files to the FastAPI backend.

---

## Step 2 — Text Extraction

The system first tries normal PDF text extraction using **PyPDF**.

If a page has little or no extractable text, an **OCR fallback** is used.

```text
Normal PDF
   ↓
PyPDF
   ↓
Text

Scanned PDF
   ↓
PyMuPDF → Page Image → Tesseract OCR
   ↓
Text
```

This allows the system to handle both normal text PDFs and scanned/image PDFs.

---

## Step 3 — Text Cleaning and Chunking

The extracted text is prepared and divided into smaller chunks.

Current configuration:

```text
Chunk size    = 1000 characters
Chunk overlap = 500 characters
```

### Why overlap?

Important information may occur near the boundary of two chunks.

The overlap helps preserve context between neighboring chunks.

---

# 4. Embeddings

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Conceptually:

```text
Text Chunk
    ↓
Embedding Model
    ↓
Numerical Vector
```

The vectors allow the system to perform **semantic search**.

This means the system can retrieve relevant content even when the wording of the question is different from the wording in the document.

---

# 5. ChromaDB — Vector Store

The embeddings and document metadata are stored in **ChromaDB**.

Current collection:

```text
document_chunks
```

Stored information includes:

```text
Chunk text
Document source
Page number
Embedding
Extraction method
```

This allows the system to retrieve both the relevant text and where that text came from.

---

# 6. Semantic Retrieval

When a user asks a question:

> What is the purpose of the CGSCR block in ProsGradNet?

the question is converted into an embedding.

ChromaDB searches for the most semantically similar document chunks.

```text
Question
   ↓
Question Embedding
   ↓
ChromaDB
   ↓
Top Candidate Chunks
```

The system retrieves multiple candidates rather than immediately trusting a single chunk.

---

# 7. Cross-Encoder Re-Ranking

The retrieved candidates are then re-ranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The Cross-Encoder evaluates the relationship between:

```text
Question + Candidate Chunk
```

and produces a relevance score.

```text
                    ┌─ Candidate 1 → Score
Question ───────────┼─ Candidate 2 → Score
                    ├─ Candidate 3 → Score
                    └─ Candidate 4 → Score
                              ↓
                       Best Evidence
```

### Important

The Cross-Encoder score is a **relevance score**, not an accuracy percentage.

---

# 8. Evidence Selection

After re-ranking, the strongest chunks are selected as evidence.

Each evidence item can contain:

```text
Evidence ID
Source document
Page number
Extraction method
Retrieval distance
Reranker score
Citation
```

Example:

```text
E1
Source: ProsGradNet.pdf
Page: 2
Extraction: TEXT
Rerank Score: ...
```

This makes the final answer traceable to the source document.

---

# 9. LLM Answer Generation

The selected evidence is passed to the LLM through the Groq API.

Current model:

```text
openai/gpt-oss-20b
```

The LLM is instructed to use the supplied evidence rather than invent unsupported information.

```text
Question
   +
Retrieved Evidence
   ↓
Prompt
   ↓
LLM
   ↓
Grounded Answer
```

---

# 10. Handling Unsupported Questions

A good RAG system should not answer every question simply because an LLM can generate an answer.

For example:

> What is the patient's blood group according to the ProsGradNet paper?

If the document does not contain blood-group information, the system should indicate that the information is not available in the retrieved evidence instead of fabricating an answer.

### Core principle

```text
Evidence available
       ↓
Generate grounded answer

Evidence unavailable
       ↓
Do not fabricate
```

This is one of the important differences between a document-grounded RAG system and a basic LLM chatbot.

---

# 11. Multi-Document Question Answering

Multiple PDFs can be indexed in the same workspace.

```text
DenseNet
    +
PA-FoV
    +
ProsGradNet
    ↓
Shared Vector Store
    ↓
User Question
    ↓
Relevant Document + Chunks
```

Example:

> Which paper discusses prostate cancer grading?

The retrieval system can identify the relevant source:

```text
ProsGradNet.pdf
```

---

# 12. Basic Consistency / Conflict Detection

The project also contains a question-aware consistency check for measurable properties.

Examples include:

```text
Accuracy
Precision
Recall
F1
Jaccard
AUC
Specificity
Loss
Parameters
Runtime
Learning Rate
Batch Size
Epochs
Temperature
Pulse
Blood Pressure
Glucose
Hemoglobin
```

When the same measurable property appears with different values across documents, the system can flag the evidence for review.

Example:

```text
Document A → Accuracy = 92.68%
Document B → Accuracy = 95.10%

              ↓

        REVIEW_REQUIRED
```

This is a consistency signal, not a claim that one document is correct.

---

# 13. Technology Stack

### Programming
- Python

### Frontend
- Streamlit

### Backend
- FastAPI
- Uvicorn
- Pydantic

### Document Processing
- PyPDF
- PyMuPDF
- Tesseract OCR
- Pillow

### RAG / NLP
- LangChain
- Sentence Transformers
- Hugging Face Embeddings
- Cross-Encoder Re-ranking

### Vector Database
- ChromaDB

### LLM
- Groq API
- `openai/gpt-oss-20b`

---

# 14. Project Structure

```text
clinical_copilot_v1/
│
├── backend/
│   └── main.py
│
├── core/
│   ├── rag.py
│   ├── ocr.py
│   └── ocr_test.py
│
├── frontend/
│   └── app.py
│
├── evaluation/
│   ├── evaluation.py
│   └── evaluate_rag.py
│
├── tests/
│   └── test_chunking.py
│
├── data/
│   └── documents/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Important files

| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI backend and API endpoints |
| `core/rag.py` | Main RAG pipeline |
| `core/ocr.py` | OCR processing for scanned PDFs |
| `frontend/app.py` | Streamlit user interface |
| `evaluation/evaluation.py` | Evaluation questions and expected sources |
| `evaluation/evaluate_rag.py` | Runs retrieval evaluation |
| `tests/test_chunking.py` | Basic chunking test |
| `requirements.txt` | Python dependencies |
| `.env.example` | Example environment configuration |
| `.gitignore` | Prevents secrets and local files from being committed |

---

# 15. API Architecture

```text
                    Streamlit
                       │
          ┌────────────┴────────────┐
          │                         │
     Upload PDFs                Ask Question
          │                         │
          ▼                         ▼
       FastAPI                  FastAPI
          │                         │
          ▼                         ▼
    RAG Ingestion             RAG Retrieval
          │                         │
          │                    Re-ranking
          │                         │
          │                         ▼
          │                    LLM Generation
          │                         │
          └──────────────┬──────────┘
                         ▼
                 Answer + Evidence
```

FastAPI also provides automatic API documentation at:

```text
/docs
```

---

# 16. Evaluation

The retrieval evaluation uses three research papers:

1. DenseNet
2. Pixel-Adaptive Field-of-View for Remote Sensing Image Segmentation
3. ProsGradNet

Evaluation set:

```text
9 answerable questions
3 unanswerable questions
```

### Retrieval results

| Metric | Result |
|---|---:|
| Recall@1 | 1.00 |
| Recall@3 | 1.00 |
| Recall@5 | 1.00 |
| MRR | 1.00 |

### Correct interpretation

On this small 9-question **document-level retrieval evaluation set**, the expected source document was retrieved in the top-1 result for all 9 answerable questions.

These results **do not mean** that the complete RAG system has 100% answer accuracy or that hallucinations are impossible.

---

# 17. Example Questions

### DenseNet

```text
What is the main connectivity idea introduced by DenseNet?
```

```text
What does the growth rate k represent in DenseNet?
```

### PA-FoV

```text
What problem is the PA-FoV module designed to solve?
```

```text
What dilation rates are used in the PA-FoV module?
```

### ProsGradNet

```text
What is the purpose of the CGSCR block in ProsGradNet?
```

```text
How many parameters does ProsGradNet have?
```

### Cross-document

```text
Which paper discusses prostate cancer grading?
```

### Unsupported

```text
What is the patient's blood group according to the ProsGradNet paper?
```

---

# 18. How to Run

## 1. Create virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

**Never commit the real API key to GitHub.**

## 4. Start FastAPI

From the project root:

```bash
uvicorn backend.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## 5. Start Streamlit

Open another terminal:

```bash
.venv\Scripts\activate
streamlit run frontend\app.py
```

Then open the Streamlit URL shown in the terminal.

---

# 19. End-to-End Example

Suppose the user uploads:

```text
DenseNet 121.pdf
Pixel-Adaptive_Field-of-View.pdf
ProsGradNet.pdf
```

Then asks:

> What accuracy does ProsGradNet achieve on the KMC Kidney dataset?

The pipeline is:

```text
User Question
      ↓
Question Embedding
      ↓
ChromaDB Retrieval
      ↓
Relevant ProsGradNet Chunks
      ↓
Cross-Encoder Re-Ranking
      ↓
Top Evidence
      ↓
Groq LLM
      ↓
Grounded Answer
      ↓
Source + Page Citation
```

The important point is that the LLM receives the relevant document evidence before generating the response.

---

# 20. Key Design Decisions

### Why RAG?

The application needs to answer questions from user-provided documents instead of relying only on the LLM's pretrained knowledge.

### Why embeddings?

Embeddings allow semantic similarity search, so relevant content can be found even when the question uses different wording.

### Why ChromaDB?

It provides the vector-store layer used to store and retrieve document embeddings.

### Why Cross-Encoder?

It provides a second-stage relevance check after initial vector retrieval.

### Why OCR?

Some PDFs are scanned images and do not contain machine-readable text.

### Why FastAPI + Streamlit?

FastAPI separates the backend/API layer from the interface, while Streamlit provides a simple interface for demonstrating the AI/ML application.

---

# 21. What This Project Demonstrates

This project demonstrates practical understanding of:

- RAG architecture
- PDF document ingestion
- Text extraction
- OCR
- Text chunking
- Embeddings
- Vector databases
- Semantic retrieval
- Cross-Encoder re-ranking
- Evidence selection
- LLM integration
- Prompt design
- Grounded question answering
- Multi-document retrieval
- Basic evaluation
- FastAPI backend development
- Streamlit frontend development
- Environment-variable and API-key handling

---

# 22. 60-Second Interview Explanation

> I built an end-to-end multi-document RAG application for question answering over PDF documents. The user can upload multiple PDFs through a Streamlit interface, while FastAPI handles the backend processing.
>
> During ingestion, the system extracts text using PyPDF, with Tesseract OCR as a fallback for scanned pages. The text is cleaned and split into overlapping chunks, which are converted into embeddings using Sentence Transformers and stored in ChromaDB.
>
> When a user asks a question, I first perform semantic retrieval to get candidate chunks. I then use a Cross-Encoder to re-rank those candidates and select the most relevant evidence. That evidence is passed to a Groq-hosted LLM, which generates a grounded answer with source and page citations.
>
> I also added handling for unsupported questions and a basic consistency check for measurable values across documents. For evaluation, I used three research papers and nine answerable questions. The expected source document was retrieved in the top-1 result for all nine questions, giving Recall@1, Recall@3, Recall@5 and MRR of 1.00 on this small evaluation set.

---

# 23. One-Line Project Summary

> **An end-to-end multi-document RAG platform that retrieves, re-ranks, and cites evidence from PDF documents before generating grounded answers using an LLM.**

---

# 24. Project Status

```text
PDF Ingestion             ✓
Text Extraction           ✓
OCR Fallback              ✓
Chunking                  ✓
Embeddings                ✓
ChromaDB                  ✓
Semantic Retrieval        ✓
Cross-Encoder Re-ranking  ✓
Evidence Generation       ✓
LLM Integration           ✓
Grounded Answers          ✓
Multi-document Support    ✓
FastAPI Backend           ✓
Streamlit Frontend        ✓
Evaluation                ✓
```

**Core project is complete and ready for GitHub packaging.**
