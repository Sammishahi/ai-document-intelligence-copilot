import os
import re
from pathlib import Path

import pymupdf
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from sentence_transformers import CrossEncoder

from .ocr import OCRProcessor


class DocumentRAG:
    """
    AI Document Intelligence Copilot RAG engine.

    Pipeline:
    PDF -> Text/OCR -> Chunks -> Embeddings
    -> ChromaDB -> Retrieval -> Reranking
    -> Evidence -> Conflict Detection
    """

    def __init__(self, model_name: str | None = None):

        # Embedding model
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Reranker
        self.reranker = CrossEncoder(
            os.getenv(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            device="cpu",
        )

        # Vector database
        self.vectorstore = Chroma(
            collection_name=os.getenv(
                "CHROMA_COLLECTION",
                "document_chunks",
            ),
            embedding_function=self.embeddings,
            persist_directory=os.getenv(
                "CHROMA_PERSIST_DIR",
                "./data/chroma",
            ),
        )

        self.ocr = OCRProcessor()

        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=500,
            length_function=len,
        )

        self.sources: set[str] = set()

    # =====================================================
    # TEXT CLEANING
    # =====================================================

    @staticmethod
    def clean(text: str) -> str:
        """Clean extracted PDF/OCR text."""

        text = text or ""
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # =====================================================
    # DELETE DOCUMENT
    # =====================================================

    def delete_source(self, source: str) -> None:
        """Replace an existing document version."""

        try:
            self.vectorstore.delete(
                where={"source": source}
            )
        except Exception as exc:
            print(
                f"[WARNING] Could not delete "
                f"'{source}': {exc}"
            )

        self.sources.discard(source)

    # =====================================================
    # PDF INGESTION
    # =====================================================

    def ingest_pdf(
        self,
        path: str,
        source_name: str | None = None,
    ) -> int:
        """
        Extract PDF text, use OCR when needed,
        create chunks and store them in ChromaDB.
        """

        pdf_path = Path(path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF documents are supported."
            )

        source = source_name or pdf_path.name

        reader = PdfReader(str(pdf_path))
        ocr_document = pymupdf.open(str(pdf_path))

        pages = []

        try:
            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):
                text = self.clean(
                    page.extract_text() or ""
                )

                extraction_method = "text"

                # OCR fallback for scanned pages
                if not text:
                    print(
                        f"[OCR] Processing page "
                        f"{page_number}"
                    )

                    text = self.clean(
                        self.ocr.extract_page_text(
                            ocr_document[
                                page_number - 1
                            ]
                        )
                    )

                    extraction_method = "ocr"

                if not text:
                    continue

                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "page": page_number,
                            "extraction_method": (
                                extraction_method
                            ),
                        },
                    )
                )

            if not pages:
                raise ValueError(
                    "No text could be extracted "
                    "from the PDF."
                )

            chunks = self.text_splitter.split_documents(
                pages
            )

            if not chunks:
                raise ValueError(
                    "No chunks were created."
                )

            # Replace old copy of the document
            self.delete_source(source)

            self.vectorstore.add_documents(chunks)
            self.sources.add(source)

            print(
                f"[INFO] Added {len(chunks)} chunks "
                f"from '{source}'"
            )

            return len(chunks)

        finally:
            ocr_document.close()

    # =====================================================
    # SOURCE FILTER
    # =====================================================

    @staticmethod
    def build_source_filter(
        sources: list[str] | None,
    ):
        """Create a ChromaDB filter for selected documents."""

        if not sources:
            return None

        sources = list(
            dict.fromkeys(
                source.strip()
                for source in sources
                if source and source.strip()
            )
        )

        if not sources:
            return None

        if len(sources) == 1:
            return {"source": sources[0]}

        return {
            "$or": [
                {"source": source}
                for source in sources
            ]
        }

    # =====================================================
    # RETRIEVAL + RERANKING
    # =====================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Two-stage retrieval:

        1. Semantic retrieval from ChromaDB
        2. Cross-Encoder reranking
        """

        query = query.strip()

        if not query:
            return []

        top_k = max(1, min(int(top_k), 10))

        # Retrieve extra candidates before reranking
        candidate_k = min(
            max(top_k * 2, 10),
            20,
        )

        search_args = {
            "k": candidate_k
        }

        chroma_filter = self.build_source_filter(
            source_filter
        )

        if chroma_filter:
            search_args["filter"] = chroma_filter

        try:
            results = (
                self.vectorstore
                .similarity_search_with_score(
                    query,
                    **search_args,
                )
            )
        except Exception as exc:
            print(
                f"[WARNING] Retrieval failed: {exc}"
            )
            return []

        if not results:
            return []

        candidates = [
            {
                "document": document,
                "distance": float(distance),
            }
            for document, distance in results
        ]

        pairs = [
            (
                query,
                item["document"].page_content,
            )
            for item in candidates
        ]

        scores = self.reranker.predict(pairs)

        for item, score in zip(
            candidates,
            scores,
        ):
            item["rerank_score"] = float(score)

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return [
            {
                "text": item["document"].page_content,
                "source": item["document"].metadata.get(
                    "source",
                    "unknown",
                ),
                "page": item["document"].metadata.get(
                    "page",
                    0,
                ),
                "extraction_method": item[
                    "document"
                ].metadata.get(
                    "extraction_method",
                    "unknown",
                ),
                "distance": round(
                    item["distance"],
                    4,
                ),
                "rerank_score": round(
                    item["rerank_score"],
                    4,
                ),
            }
            for item in candidates[:top_k]
        ]

    # =====================================================
    # EVIDENCE
    # =====================================================

    def build_evidence(
        self,
        query: str,
        top_k: int = 5,
        source_filter: list[str] | None = None,
    ) -> list[dict]:

        retrieved = self.retrieve(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
        )

        return [
            {
                "evidence_id": f"E{index}",
                "text": item["text"],
                "source": item["source"],
                "page": item["page"],
                "extraction_method": item[
                    "extraction_method"
                ],
                "distance": item["distance"],
                "rerank_score": item[
                    "rerank_score"
                ],
                "citation": (
                    f"{item['source']}, "
                    f"page {item['page']}"
                ),
            }
            for index, item in enumerate(
                retrieved,
                start=1,
            )
        ]

    # =====================================================
    # CONFLICT DETECTION
    # =====================================================

    def build_conflict_ledger(
        self,
        evidence: list[dict],
        query: str = "",
    ) -> dict:
        """
        Detect possible inconsistencies only for the
        measurable property asked about in the question.

        Different values across documents are marked
        REVIEW_REQUIRED.
        """

        properties = {
            "accuracy": [
                "accuracy",
                "classification accuracy",
            ],
            "precision": ["precision"],
            "recall": [
                "recall",
                "sensitivity",
            ],
            "f1_score": [
                "f1",
                "f1 score",
                "f1-score",
            ],
            "jaccard": [
                "jaccard",
                "jaccard index",
            ],
            "auc": [
                "auc",
                "area under the curve",
            ],
            "specificity": ["specificity"],
            "loss": [
                "loss",
                "validation loss",
                "training loss",
            ],
            "parameters": [
                "parameters",
                "parameter count",
                "number of parameters",
            ],
            "flops": ["flops"],
            "runtime": [
                "runtime",
                "inference time",
                "training time",
            ],
            "learning_rate": [
                "learning rate"
            ],
            "batch_size": ["batch size"],
            "epochs": [
                "epochs",
                "number of epochs",
            ],
            "temperature": ["temperature"],
            "pulse": [
                "pulse",
                "heart rate",
            ],
            "blood_pressure": [
                "blood pressure",
                "bp",
            ],
            "glucose": [
                "glucose",
                "blood glucose",
            ],
            "hemoglobin": [
                "hemoglobin",
                "haemoglobin",
                "hb",
            ],
        }

        query_lower = query.lower()

        # Identify the property asked about
        target = next(
            (
                name
                for name, aliases in properties.items()
                if any(
                    re.search(
                        rf"\b{re.escape(alias)}\b",
                        query_lower,
                    )
                    for alias in aliases
                )
            ),
            None,
        )

        # General/non-numeric questions
        if not target:
            return {
                "has_conflict": False,
                "status": "NO_CONFLICT_DETECTED",
                "conflict_count": 0,
                "conflicts": [],
                "detection_method": (
                    "Question-aware property comparison"
                ),
            }

        aliases = properties[target]

        # Numeric values such as:
        # 92.68%, 0.34M, 98.4 F, 128/82
        value_pattern = re.compile(
            r"\d+(?:\.\d+)?"
            r"(?:\s*/\s*\d+(?:\.\d+)?)?"
            r"(?:\s*[KMB])?"
            r"(?:\s*[%°])?"
            r"(?:\s*[CF])?",
            re.IGNORECASE,
        )

        statements = []

        for item in evidence:

            text = item.get("text", "")

            for sentence in re.split(
                r"(?<=[.!?])\s+|\n+",
                text,
            ):
                sentence = sentence.strip()

                if not sentence:
                    continue

                sentence_lower = sentence.lower()

                match = next(
                    (
                        re.search(
                            rf"\b{re.escape(alias)}\b",
                            sentence_lower,
                        )
                        for alias in aliases
                        if re.search(
                            rf"\b{re.escape(alias)}\b",
                            sentence_lower,
                        )
                    ),
                    None,
                )

                if not match:
                    continue

                start = max(
                    0,
                    match.start() - 60,
                )

                end = min(
                    len(sentence),
                    match.end() + 60,
                )

                nearby_text = sentence[start:end]

                values = list(
                    value_pattern.finditer(
                        nearby_text
                    )
                )

                if not values:
                    continue

                value = min(
                    values,
                    key=lambda value_match: abs(
                        value_match.start()
                        - (match.start() - start)
                    ),
                ).group(0)

                statements.append(
                    {
                        "property": target,
                        "value": (
                            value
                            .strip()
                            .replace(" ", "")
                            .lower()
                        ),
                        "text": sentence,
                        "source": item.get(
                            "source",
                            "unknown",
                        ),
                        "page": item.get(
                            "page",
                            0,
                        ),
                        "evidence_id": item.get(
                            "evidence_id",
                            "unknown",
                        ),
                    }
                )

        # Remove duplicates
        unique = []
        seen = set()

        for statement in statements:

            key = (
                statement["property"],
                statement["source"],
                statement["page"],
                statement["value"],
                statement["text"].lower(),
            )

            if key not in seen:
                seen.add(key)
                unique.append(statement)

        # Compare values only across different documents
        conflicts = []

        for index, first in enumerate(unique):

            for second in unique[index + 1:]:

                if (
                    first["property"]
                    != second["property"]
                ):
                    continue

                if (
                    first["source"]
                    == second["source"]
                ):
                    continue

                if (
                    first["value"]
                    == second["value"]
                ):
                    continue

                conflicts.append(
                    {
                        "type": "potential_inconsistency",
                        "property": first[
                            "property"
                        ],
                        "statement_1": {
                            "text": first["text"],
                            "values": [
                                first["value"]
                            ],
                            "source": first["source"],
                            "page": first["page"],
                            "evidence_id": first[
                                "evidence_id"
                            ],
                        },
                        "statement_2": {
                            "text": second["text"],
                            "values": [
                                second["value"]
                            ],
                            "source": second["source"],
                            "page": second["page"],
                            "evidence_id": second[
                                "evidence_id"
                            ],
                        },
                        "status": "REVIEW_REQUIRED",
                        "message": (
                            "Different values were "
                            "detected for the same "
                            "property across documents. "
                            "The system does not "
                            "determine which value is "
                            "authoritative."
                        ),
                    }
                )

        return {
            "has_conflict": bool(conflicts),
            "status": (
                "CONFLICT_DETECTED"
                if conflicts
                else "NO_CONFLICT_DETECTED"
            ),
            "conflict_count": len(conflicts),
            "conflicts": conflicts,
            "detection_method": (
                "Question-aware cross-document "
                "property comparison"
            ),
        }

    # =====================================================
    # COMPLETE EVIDENCE PACKAGE
    # =====================================================

    def get_evidence_package(
        self,
        query: str,
        top_k: int = 5,
        source_filter: list[str] | None = None,
    ) -> dict:

        evidence = self.build_evidence(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
        )

        conflicts = self.build_conflict_ledger(
            evidence=evidence,
            query=query,
        )

        return {
            "query": query,
            "evidence": evidence,
            "conflict_ledger": conflicts,
            "selected_documents": source_filter,
        }