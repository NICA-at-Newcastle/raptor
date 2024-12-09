"""Tree builder tests."""

import pytest


@pytest.mark.slow
def test_builds_tree(
    memory_storage,
    snapshot,
):
    """Test that raptor builds the tree from the cinderella text."""

    assert memory_storage.to_dict() == snapshot
