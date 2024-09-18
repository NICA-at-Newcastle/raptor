"""Tree builder tests."""

import pytest


@pytest.mark.slow
def test_builds_tree(tree_builder, cinderella_text, memory_storage, snapshot):
    """Test that raptor builds the tree from the cinderella text."""

    chunks = cinderella_text.split(".")

    _ = tree_builder.build_from_chunks(chunks)

    assert snapshot == memory_storage.to_dict()
