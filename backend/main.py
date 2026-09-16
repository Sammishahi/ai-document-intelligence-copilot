import os
import tempfile
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from core.rag import  DocumentRAG


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AI Document Intelligence Copilot API",
    version="0.3.0",
    description=(
        "Evidence-grounded multi-document intelligence "
        "and question-answering API with document-level "
        "retrieval filtering."
    ),
)


# =========================================================
# RAG ENGINE
# =========================================================

rag = DocumentRAG()


# =========================================================
# REQUEST MODEL
# =========================================================

class QueryRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Question to answer from the indexed "
            "document workspace."
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "Number of evidence chunks to retrieve."
        ),
    )

    selected_documents: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of document filenames to "
            "restrict retrieval to. If omitted, all "
            "currently indexed documents are searchable."
        ),
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health() -> dict[str, Any]:

    return {
        "status": "ok",

        "documents_indexed": len(
            rag.sources
        ),

        "chunks_indexed": (
            rag.vectorstore
            ._collection
            .count()
        ),
    }


# =========================================================
# DOCUMENT UPLOAD
# =========================================================

@app.post("/documents")
async def upload_documents(
    files: list[UploadFile] = File(...)
) -> dict[str, Any]:

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if not files:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload at least one PDF document."
            ),
        )

    selected_sources = []

    for file in files:

        filename = (
            file.filename
            or "document.pdf"
        )

        if not filename.lower().endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Only PDF files are supported: "
                    f"{filename}"
                ),
            )

        selected_sources.append(
            filename
        )

    # -----------------------------------------------------
    # Remove duplicate filenames
    # -----------------------------------------------------

    selected_sources = list(
        dict.fromkeys(
            selected_sources
        )
    )

    # -----------------------------------------------------
    # WORKSPACE ISOLATION
    #
    # Only documents selected in the current upload
    # request remain searchable.
    # -----------------------------------------------------

    try:

        rag.vectorstore.delete(
            where={
                "source": {
                    "$nin": selected_sources
                }
            }
        )

    except Exception as exc:

        print(
            "[WARNING] Could not clean old "
            f"workspace documents: {exc}"
        )

    # Reset runtime source tracking

    rag.sources.clear()

    # -----------------------------------------------------
    # Process uploaded documents
    # -----------------------------------------------------

    results = []

    total_chunks = 0
    successful_files = 0
    failed_files = 0

    for file in files:

        filename = (
            file.filename
            or "document.pdf"
        )

        content = await file.read()

        if not content:

            failed_files += 1

            results.append(
                {
                    "filename": filename,
                    "status": "failed",
                    "error": "Uploaded file is empty.",
                }
            )

            continue

        temp_path = None

        try:

            # ---------------------------------------------
            # Save temporary PDF
            # ---------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as tmp:

                tmp.write(content)

                temp_path = tmp.name

            # ---------------------------------------------
            # Ingest document
            # ---------------------------------------------

            chunks_added = rag.ingest_pdf(
                temp_path,
                source_name=filename,
            )

            total_chunks += chunks_added

            successful_files += 1

            results.append(
                {
                    "filename": filename,
                    "status": "success",
                    "chunks_added": chunks_added,
                }
            )

        except ValueError as exc:

            failed_files += 1

            results.append(
                {
                    "filename": filename,
                    "status": "failed",
                    "error": str(exc),
                }
            )

        except Exception as exc:

            failed_files += 1

            results.append(
                {
                    "filename": filename,
                    "status": "failed",
                    "error": str(exc),
                }
            )

        finally:

            if temp_path:

                try:

                    os.remove(
                        temp_path
                    )

                except OSError:

                    pass

    # -----------------------------------------------------
    # Validate processing result
    # -----------------------------------------------------

    if successful_files == 0:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "None of the uploaded documents "
                    "could be processed."
                ),

                "results": results,
            },
        )

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    return {
        "status": "success",

        "total_files": len(files),

        "successful_files": (
            successful_files
        ),

        "failed_files": (
            failed_files
        ),

        "total_chunks": (
            rag.vectorstore
            ._collection
            .count()
        ),

        "documents": [
            item["filename"]
            for item in results
            if item["status"] == "success"
        ],

        "results": results,
    }


# =========================================================
# QUESTION ANSWERING
# =========================================================

@app.post("/query")
def query(
    req: QueryRequest
) -> dict[str, Any]:

    question = req.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # -----------------------------------------------------
    # Clean document selection
    # -----------------------------------------------------

    selected_documents = None

    if req.selected_documents:

        selected_documents = list(
            dict.fromkeys(
                filename.strip()
                for filename in req.selected_documents
                if filename.strip()
            )
        )

        if not selected_documents:

            selected_documents = None

    # -----------------------------------------------------
    # Retrieve evidence
    #
    # selected_documents will be passed to the RAG engine.
    # The RAG engine will apply this list as a ChromaDB
    # metadata filter.
    # -----------------------------------------------------

    evidence_package = (
        rag.get_evidence_package(
            question,
            req.top_k,
            source_filter=selected_documents,
        )
    )

    evidence = (
        evidence_package.get(
            "evidence",
            []
        )
    )

    conflict_ledger = (
        evidence_package.get(
            "conflict_ledger",
            {}
        )
    )

    # -----------------------------------------------------
    # No evidence
    # -----------------------------------------------------

    if not evidence:

        if selected_documents:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No relevant evidence was found "
                    "within the selected documents."
                ),
            )

        raise HTTPException(
            status_code=404,
            detail=(
                "No indexed evidence found. "
                "Upload and process documents first."
            ),
        )

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = build_context(
        evidence
    )

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    answer = generate_answer(
        question=question,
        context=context,
        conflict_ledger=conflict_ledger,
    )

    # -----------------------------------------------------
    # Citations
    # -----------------------------------------------------

    citations = build_citations(
        evidence
    )

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    return {
        "question": question,

        "selected_documents": (
            selected_documents
        ),

        "answer": answer,

        "evidence": evidence,

        "citations": citations,

        "conflict_ledger": conflict_ledger,
    }


# =========================================================
# CONTEXT BUILDER
# =========================================================

def build_context(
    evidence: list[dict]
) -> str:

    context_blocks = []

    for item in evidence:

        evidence_id = item.get(
            "evidence_id",
            "UNKNOWN",
        )

        source = item.get(
            "source",
            "unknown",
        )

        page = item.get(
            "page",
            "unknown",
        )

        extraction_method = item.get(
            "extraction_method",
            "unknown",
        )

        text = item.get(
            "text",
            "",
        )

        rerank_score = item.get(
            "rerank_score",
            "unknown",
        )

        context_blocks.append(
            f"""
[{evidence_id}]
Source: {source}
Page: {page}
Extraction method: {extraction_method}
Rerank score: {rerank_score}

Evidence:
{text}
""".strip()
        )

    return "\n\n".join(
        context_blocks
    )


# =========================================================
# CITATION BUILDER
# =========================================================

def build_citations(
    evidence: list[dict]
) -> list[dict]:

    citations = []

    for item in evidence:

        citations.append(
            {
                "evidence_id": item.get(
                    "evidence_id"
                ),

                "source": item.get(
                    "source"
                ),

                "page": item.get(
                    "page"
                ),

                "citation": item.get(
                    "citation"
                ),

                "extraction_method": item.get(
                    "extraction_method"
                ),

                "rerank_score": item.get(
                    "rerank_score"
                ),
            }
        )

    return citations


# =========================================================
# LLM ANSWER GENERATION
# =========================================================

def generate_answer(
    question: str,
    context: str,
    conflict_ledger: dict,
) -> str:

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        return (
            "GROQ_API_KEY is not configured. "
            "Please configure the API key in the .env file."
        )

    # -----------------------------------------------------
    # LLM
    # -----------------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=api_key,
    )

    # -----------------------------------------------------
    # System Prompt
    # -----------------------------------------------------

    system_prompt = """
You are an evidence-grounded document information assistant.

Your task is to answer questions using ONLY the
document evidence supplied to you.

You are not allowed to invent information.

=========================================================
GROUNDING RULES
=========================================================

1. Use ONLY the supplied document evidence.

2. Do NOT use outside knowledge to fill missing
   document information.

3. Do NOT invent:

   - facts
   - numbers
   - dates
   - names
   - measurements
   - results
   - diagnoses
   - conclusions
   - procedures
   - methodologies

4. If the retrieved evidence does not contain enough
   information to answer the question, explicitly state:

   "The available documents do not provide enough
   information."

5. Every important factual statement should reference
   the relevant evidence ID.

   Example:

   "The model used Adam as the optimizer [E2]."

6. Clearly distinguish documented facts from interpretation.

7. Do not treat your own inference as a documented fact.

=========================================================
DOCUMENT FILTERING
=========================================================

The retrieved evidence may have been restricted to
specific documents selected by the user.

Use ONLY the evidence supplied in the current context.

Do not introduce information from documents that are
not represented in the supplied evidence.

=========================================================
MULTI-DOCUMENT RULES
=========================================================

1. Multiple documents may be supplied.

2. Identify the source document relevant to the question.

3. Do not mix information from unrelated documents.

4. If multiple documents contain relevant information,
   clearly identify which source supports each statement.

5. If documents provide different values for the same
   property and the conflict ledger reports a potential
   inconsistency:

   - Do NOT choose one value.
   - Do NOT silently merge the values.
   - Report the difference.
   - Cite the relevant evidence IDs.
   - State that document/source review is required.

=========================================================
CONFLICT HANDLING
=========================================================

A conflict detected by the system is a POTENTIAL
inconsistency, not proof that the document is wrong.

Example:

"Different accuracy values were detected in the
retrieved evidence [E1, E4]. The system cannot
determine which value is authoritative."

=========================================================
ANSWER STYLE
=========================================================

Keep answers concise and evidence-grounded.

Prefer:

Answer
Evidence / explanation
Sources

Do not add unnecessary speculation.
"""

    conflict_text = str(
        conflict_ledger
    )

    # -----------------------------------------------------
    # User Prompt
    # -----------------------------------------------------

    user_prompt = f"""
QUESTION
========

{question}


RETRIEVED DOCUMENT EVIDENCE
===========================

{context}


CONFLICT LEDGER
===============

{conflict_text}


TASK
====

Answer the question using ONLY the retrieved document
evidence.

Use evidence IDs such as [E1], [E2], etc.

If the evidence is insufficient, say:

"The available documents do not provide enough
information."

If multiple potentially conflicting values are present,
report the conflict and identify the relevant evidence IDs.

Do not invent information.
"""

    # -----------------------------------------------------
    # Generate response
    # -----------------------------------------------------

    try:

        response = llm.invoke(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    user_prompt,
                ),
            ]
        )

        return str(
            response.content
        ).strip()

    except Exception as exc:

        return (
            "The document evidence was retrieved "
            "successfully, but answer generation "
            "could not be completed: "
            f"{exc}"
        )