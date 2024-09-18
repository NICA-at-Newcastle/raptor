"""Tree retriever tests"""


def test_retreives_text(cinderella_tree, tree_retriever, memory_storage, snapshot):
    """Test that tree raptor retrieves from pre-computed cinderella tree."""

    memory_storage.load_dict(cinderella_tree)
    results = tree_retriever.retrieve(
        "What is cinderella doing?",
        tree_retriever.SearchMethod.Tree(),
    )

    returned_chunks = [result["chunk"] for result in results]
    assert snapshot == returned_chunks
