import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "evaluation"))

from core.rag import DocumentRAG
from evaluation import EVALUATION_DATA


def source_matches(expected, actual):
    """Check whether retrieved document is the expected document."""
    expected = expected.lower()
    actual = actual.lower()

    return expected in actual


def main():

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    rag = DocumentRAG()

    answerable = [
        item for item in EVALUATION_DATA
        if item["answerable"]
    ]

    for k in [1, 3, 5]:

        correct = 0

        for item in answerable:

            results = rag.retrieve(
                query=item["question"],
                top_k=5
            )

            sources = [
                result.get("source", "")
                for result in results
            ]

            if any(
                source_matches(
                    item["expected_source"],
                    source
                )
                for source in sources[:k]
            ):
                correct += 1

        score = correct / len(answerable)

        print(f"Recall@{k}: {score:.2f}")

    # -------------------------
    # MRR
    # -------------------------

    reciprocal_ranks = []

    for item in answerable:

        results = rag.retrieve(
            query=item["question"],
            top_k=5
        )

        sources = [
            result.get("source", "")
            for result in results
        ]

        rank_found = None

        for rank, source in enumerate(
            sources,
            start=1
        ):
            if source_matches(
                item["expected_source"],
                source
            ):
                rank_found = rank
                break

        if rank_found:
            reciprocal_ranks.append(
                1 / rank_found
            )
        else:
            reciprocal_ranks.append(0)

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    print(f"MRR: {mrr:.2f}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()