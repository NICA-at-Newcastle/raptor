"""Tree retriever tests"""

from raptor import TreeRetriever
from raptor.embedding_models import SBertEmbeddingModel


def test_retreives_text_with_outliers(memory_storage, snapshot):
    """Test that tree raptor retrieves from pre-computed cinderella tree."""

    sbert = SBertEmbeddingModel()

    tree_retriever = TreeRetriever(
        TreeRetriever.Config(
            storage=memory_storage,
            embedding_model=sbert,
            limit=TreeRetriever.Limit.TopK(5),
        )
    )

    results = tree_retriever.retrieve(
        "What is cinderella doing?",
        tree_retriever.SearchMethod.Tree((-1, True)),
    )

    returned_chunks = [result["chunk"] for result in results]
    assert snapshot == returned_chunks


def test_retreives_text_no_outliers(memory_storage, snapshot):
    """Test that tree raptor retrieves from pre-computed cinderella tree."""

    sbert = SBertEmbeddingModel()

    tree_retriever = TreeRetriever(
        TreeRetriever.Config(
            storage=memory_storage,
            embedding_model=sbert,
            limit=TreeRetriever.Limit.TopK(5),
        )
    )

    results = tree_retriever.retrieve(
        "What is cinderella doing?",
        tree_retriever.SearchMethod.Tree((-1, False)),
    )

    returned_chunks = [result["chunk"] for result in results]
    assert snapshot == returned_chunks
