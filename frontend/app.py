import re
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Document Intelligence Copilot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background: #f6f8fc;
}

.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.2rem;
}


/* ================= HERO ================= */

.hero {
    background: linear-gradient(135deg, #ffffff, #eef4ff);
    border: 1px solid #dfe7f2;
    border-radius: 18px;
    padding: 30px 32px;
    margin-bottom: 28px;
    box-shadow: 0 5px 20px rgba(15, 23, 42, 0.05);
}

.hero-title {
    font-size: 32px;
    font-weight: 750;
    color: #111827;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 15px;
    color: #64748b;
    line-height: 1.6;
}


/* ================= SECTION ================= */

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    margin-top: 20px;
    margin-bottom: 5px;
}

.section-description {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 18px;
}


/* ================= CARDS ================= */

.info-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.answer-card {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 16px;
    padding: 24px 26px;
    margin-top: 10px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
}

.answer-label {
    font-size: 12px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
}


/* ================= RERANK SCORE ================= */

.score-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 12px;
}

.score-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.score-value {
    font-size: 16px;
    font-weight: 700;
    color: #111827;
}


/* ================= BUTTONS ================= */

.stButton > button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 42px;
}


/* ================= METRICS ================= */

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid #e5e7eb;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_duplicate_citations(text: str) -> str:
    """
    Removes accidental consecutive duplicate evidence
    citations such as:

        [E1] [E1]

    while preserving normal citations.
    """

    if not text:
        return text

    pattern = re.compile(
        r"\[(E\d+)\]\s*\[\1\]"
    )

    previous = None

    while previous != text:

        previous = text

        text = pattern.sub(
            r"[\1]",
            text
        )

    return text


# =========================================================
# HERO
# =========================================================

st.markdown(
    """<div class="hero">
<div class="hero-title">📄 AI Document Intelligence Copilot</div>
<div class="hero-subtitle">
Multi-document RAG platform for evidence-grounded question answering,
semantic document retrieval, source citations and document analysis.
</div>
</div>""",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """<div style="
        font-size:22px;
        font-weight:750;
        color:#111827;
        margin-bottom:5px;
        ">Document Workspace</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div style="
        font-size:13px;
        color:#64748b;
        margin-bottom:20px;
        ">Upload and analyze your PDF documents</div>""",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF documents at once.",
    )

    if uploaded_files:

        st.markdown(
            f"""<div style="
            margin-top:10px;
            margin-bottom:12px;
            font-size:13px;
            font-weight:650;
            color:#475569;
            ">{len(uploaded_files)} document(s) selected</div>""",
            unsafe_allow_html=True,
        )

        for file in uploaded_files:

            st.markdown(
                f"""<div style="
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:9px;
                padding:9px 11px;
                margin-bottom:7px;
                font-size:13px;
                color:#334155;
                ">📄 {file.name}</div>""",
                unsafe_allow_html=True,
            )

    process_documents = st.button(
        "⚡ Process Documents",
        use_container_width=True,
    )


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

if process_documents:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF document."
        )

    else:

        files = []

        for file in uploaded_files:

            files.append(
                (
                    "files",
                    (
                        file.name,
                        file.getvalue(),
                        "application/pdf",
                    ),
                )
            )

        with st.spinner(
            "Processing documents and building the knowledge base..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/documents",
                    files=files,
                    timeout=300,
                )

                if response.status_code == 200:

                    result = response.json()

                    st.session_state["upload_result"] = result

                    # Clear previous query because the
                    # knowledge base has changed.
                    st.session_state.pop(
                        "query_result",
                        None
                    )

                    st.success(
                        "Documents processed successfully."
                    )

                else:

                    st.error(
                        f"Document processing failed: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to FastAPI backend. "
                    "Please make sure the backend server is running."
                )

            except Exception as exc:

                st.error(
                    f"An unexpected error occurred: {exc}"
                )


# =========================================================
# KNOWLEDGE BASE
# =========================================================

if "upload_result" in st.session_state:

    result = st.session_state["upload_result"]

    st.markdown(
        """<div class="section-title">Knowledge Base</div>
<div class="section-description">
Documents processed and indexed for semantic retrieval.
</div>""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documents",
        result.get("total_files", 0),
    )

    col2.metric(
        "Processed",
        result.get("successful_files", 0),
    )

    col3.metric(
        "Failed",
        result.get("failed_files", 0),
    )

    st.markdown(
        f"""<div class="info-card">
<b>Vector Database</b><br>
<span style="color:#64748b;">
{result.get("total_chunks", 0)}
document chunks indexed in ChromaDB
</span>
</div>""",
        unsafe_allow_html=True,
    )

    for item in result.get("results", []):

        if item["status"] == "success":

            st.success(
                f"✓ {item['filename']}  •  "
                f"{item['chunks_added']} chunks indexed"
            )

        else:

            st.error(
                f"✕ {item['filename']}  •  "
                f"{item.get('error', 'Unknown error')}"
            )


# =========================================================
# QUESTION SECTION
# =========================================================

st.markdown(
    """<div class="section-title">Ask Your Documents</div>
<div class="section-description">
Ask questions and receive answers grounded in the indexed
document evidence.
</div>""",
    unsafe_allow_html=True,
)


question = st.text_area(
    "Question",
    placeholder="Example: What optimizer was used to train RRCGNet?",
    height=90,
    label_visibility="collapsed",
)


top_k = st.slider(
    "Evidence chunks to retrieve",
    min_value=1,
    max_value=10,
    value=5,
)


ask_question = st.button(
    "🔍  Ask Question",
    use_container_width=True,
)


# =========================================================
# QUERY
# =========================================================

if ask_question:

    if not question.strip():

        st.warning(
            "Please enter a question before searching."
        )

    else:

        payload = {
            "question": question.strip(),
            "top_k": top_k,
        }

        with st.spinner(
            "Searching documents and generating an evidence-grounded answer..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/query",
                    json=payload,
                    timeout=120,
                )

                if response.status_code == 200:

                    result = response.json()

                    st.session_state["query_result"] = result

                elif response.status_code == 404:

                    st.warning(
                        "No indexed evidence found. "
                        "Please process your documents first."
                    )

                else:

                    st.error(
                        f"Query failed: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to FastAPI backend."
                )

            except Exception as exc:

                st.error(
                    f"An unexpected error occurred: {exc}"
                )


# =========================================================
# ANSWER
# =========================================================

if "query_result" in st.session_state:

    result = st.session_state["query_result"]

    st.markdown(
        """<div class="section-title">Answer</div>""",
        unsafe_allow_html=True,
    )

    answer = result.get(
        "answer",
        "No answer generated.",
    )

    # Remove accidental consecutive duplicate
    # evidence citations such as [E1] [E1].
    answer = clean_duplicate_citations(
        answer
    )

    st.markdown(
        '<div class="answer-card">'
        '<div class="answer-label">'
        'Evidence-Grounded Response'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(answer)

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    # =====================================================
    # EVIDENCE
    # =====================================================

    st.markdown(
        """<div class="section-title">Retrieved Evidence</div>
<div class="section-description">
The most relevant document chunks retrieved for this question
after semantic retrieval and Cross-Encoder reranking.
</div>""",
        unsafe_allow_html=True,
    )

    evidence = result.get(
        "evidence",
        [],
    )

    for item in evidence:

        evidence_id = item.get(
            "evidence_id",
            "Unknown",
        )

        source = item.get(
            "source",
            "Unknown",
        )

        page = item.get(
            "page",
            "Unknown",
        )

        extraction_method = item.get(
            "extraction_method",
            "Unknown",
        )

        distance = item.get(
            "distance",
            "Unknown",
        )

        rerank_score = item.get(
            "rerank_score",
            None,
        )

        with st.expander(
            f"📌 {evidence_id}  |  {source}  |  Page {page}"
        ):

            st.write(
                item.get(
                    "text",
                    "",
                )
            )

            # ---------------------------------------------
            # Retrieval metadata
            # ---------------------------------------------

            col1, col2, col3 = st.columns(3)

            col1.markdown(
                f"""<div class="score-card">
<div class="score-label">Rerank Score</div>
<div class="score-value">
{rerank_score if rerank_score is not None else "N/A"}
</div>
</div>""",
                unsafe_allow_html=True,
            )

            col2.markdown(
                f"""<div class="score-card">
<div class="score-label">Vector Distance</div>
<div class="score-value">
{distance}
</div>
</div>""",
                unsafe_allow_html=True,
            )

            col3.markdown(
                f"""<div class="score-card">
<div class="score-label">Extraction</div>
<div class="score-value">
{str(extraction_method).upper()}
</div>
</div>""",
                unsafe_allow_html=True,
            )


    # =====================================================
    # SOURCES
    # =====================================================

    st.markdown(
        """<div class="section-title">Sources & Citations</div>""",
        unsafe_allow_html=True,
    )

    citations = result.get(
        "citations",
        [],
    )

    for citation in citations:

        citation_id = citation.get(
            "evidence_id",
            "Unknown",
        )

        citation_text = citation.get(
            "citation",
            "Unknown source",
        )

        extraction_method = citation.get(
            "extraction_method",
            "Unknown",
        )

        st.markdown(
            f"""<div class="info-card">
<b>[{citation_id}]</b>
<span style="color:#475569;">
{citation_text}
</span>
<br>
<span style="font-size:12px;color:#94a3b8;">
Extraction: {extraction_method}
</span>
</div>""",
            unsafe_allow_html=True,
        )


    # =====================================================
    # CONFLICT LEDGER
    # =====================================================

    conflict = result.get(
        "conflict_ledger",
        {},
    )

    if conflict.get(
        "has_conflict",
        False,
    ):

        conflict_count = conflict.get(
            "conflict_count",
            0,
        )

        st.warning(
            f"⚠️ Potential inconsistency detected: "
            f"{conflict_count}"
        )

        st.caption(
            "The system detected different values for "
            "the same measurable property. This does "
            "not automatically mean the document "
            "contains an actual contradiction."
        )

        for index, item in enumerate(
            conflict.get(
                "conflicts",
                [],
            ),
            start=1,
        ):

            property_name = item.get(
                "property",
                "Unknown property",
            )

            statement_1 = item.get(
                "statement_1",
                {},
            )

            statement_2 = item.get(
                "statement_2",
                {},
            )

            st.markdown(
                f"### Conflict {index}: `{property_name}`"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "**Value 1**"
                )

                st.info(
                    ", ".join(
                        map(
                            str,
                            statement_1.get(
                                "values",
                                []
                            )
                        )
                    )
                )

                st.caption(
                    f"Evidence: "
                    f"{statement_1.get('evidence_id', 'Unknown')}"
                )

                st.caption(
                    f"Source: "
                    f"{statement_1.get('source', 'Unknown')}"
                )

                st.caption(
                    f"Page: "
                    f"{statement_1.get('page', 'Unknown')}"
                )

                with st.expander(
                    "View statement 1"
                ):

                    st.write(
                        statement_1.get(
                            "text",
                            "",
                        )
                    )

            with col2:

                st.markdown(
                    "**Value 2**"
                )

                st.info(
                    ", ".join(
                        map(
                            str,
                            statement_2.get(
                                "values",
                                []
                            )
                        )
                    )
                )

                st.caption(
                    f"Evidence: "
                    f"{statement_2.get('evidence_id', 'Unknown')}"
                )

                st.caption(
                    f"Source: "
                    f"{statement_2.get('source', 'Unknown')}"
                )

                st.caption(
                    f"Page: "
                    f"{statement_2.get('page', 'Unknown')}"
                )

                with st.expander(
                    "View statement 2"
                ):

                    st.write(
                        statement_2.get(
                            "text",
                            "",
                        )
                    )

            st.caption(
                "⚠️ Review both source statements. "
                "The system does not determine which "
                "value is authoritative."
            )

            st.divider()

    else:

        st.success(
            "✓ No potential numeric inconsistency "
            "detected in the retrieved evidence."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """<div class="footer">
AI Document Intelligence Copilot
&nbsp;•&nbsp;
RAG + ChromaDB + Cross-Encoder Reranking + FastAPI
</div>""",
    unsafe_allow_html=True,
)