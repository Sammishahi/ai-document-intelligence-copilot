from core.rag import ClinicalRAG


def test_chunking_overlap_and_content():
    text = "word " * 1000
    chunks = ClinicalRAG.chunk_text(text, size=1000, overlap=200)

    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_chunking_empty_text():
    assert ClinicalRAG.chunk_text("   ") == []


def test_invalid_overlap():
    try:
        ClinicalRAG.chunk_text("hello world", size=100, overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass
