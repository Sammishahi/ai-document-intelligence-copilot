# evaluation/evaluation.py

EVALUATION_DATA = [

    # =====================================================
    # DenseNet Paper
    # =====================================================

    {
        "question": "What is the main connectivity idea introduced by DenseNet?",
        "expected_source": "DenseNet",
        "answerable": True,
    },

    {
        "question": "What does the growth rate k represent in DenseNet?",
        "expected_source": "DenseNet",
        "answerable": True,
    },

    {
        "question": "What is the purpose of the bottleneck layer in DenseNet-B?",
        "expected_source": "DenseNet",
        "answerable": True,
    },

    {
        "question": "What is the patient's blood group according to the DenseNet paper?",
        "expected_source": "DenseNet",
        "answerable": False,
    },


    # =====================================================
    # PA-FoV Paper
    # =====================================================

    {
        "question": "What problem is the PA-FoV module designed to solve?",
        "expected_source": "Pixel-Adaptive",
        "answerable": True,
    },

    {
        "question": "What dilation rates are used in the PA-FoV module?",
        "expected_source": "Pixel-Adaptive",
        "answerable": True,
    },

    {
        "question": "How does PA-FoV generate pixel-wise weights?",
        "expected_source": "Pixel-Adaptive",
        "answerable": True,
    },

    {
        "question": "What is the patient's hemoglobin level according to the PA-FoV paper?",
        "expected_source": "Pixel-Adaptive",
        "answerable": False,
    },


    # =====================================================
    # ProsGradNet Paper
    # =====================================================

    {
        "question": "What is the main purpose of the CGSCR block in ProsGradNet?",
        "expected_source": "ProsGradNet",
        "answerable": True,
    },

    {
        "question": "What group sizes are used in the CGSCR block?",
        "expected_source": "ProsGradNet",
        "answerable": True,
    },

    {
        "question": "What accuracy does ProsGradNet achieve on the KMC Kidney dataset?",
        "expected_source": "ProsGradNet",
        "answerable": True,
    },

    {
        "question": "What is the patient's blood pressure according to the ProsGradNet paper?",
        "expected_source": "ProsGradNet",
        "answerable": False,
    },
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_answerable_questions():
    """Return only questions whose answers exist in the documents."""
    return [
        item
        for item in EVALUATION_DATA
        if item["answerable"]
    ]


def get_unanswerable_questions():
    """Return questions whose answers should not exist."""
    return [
        item
        for item in EVALUATION_DATA
        if not item["answerable"]
    ]


def get_questions_by_source(source_keyword):
    """Return all questions belonging to a document."""
    return [
        item
        for item in EVALUATION_DATA
        if source_keyword.lower()
        in item["expected_source"].lower()
    ]


if __name__ == "__main__":

    print("=" * 60)
    print("RAG EVALUATION DATASET")
    print("=" * 60)

    print(
        f"Total questions: {len(EVALUATION_DATA)}"
    )

    print(
        f"Answerable questions: "
        f"{len(get_answerable_questions())}"
    )

    print(
        f"Unanswerable questions: "
        f"{len(get_unanswerable_questions())}"
    )

    print("\nQuestions by document:")

    for source in [
        "DenseNet",
        "Pixel-Adaptive",
        "ProsGradNet",
    ]:
        questions = get_questions_by_source(source)

        print(
            f"\n{source}: {len(questions)} questions"
        )