import json
import pytest
import numpy as np
from raptor import ClusterTreeBuilder, TreeRetriever, Tree
from raptor.summarization_models import BartSummarizationModel
from raptor.embedding_models import SBertEmbeddingModel
from raptor.storages.memory_storage import MemoryStorage


@pytest.fixture
def bart_summarization_model():
    """Bart summarizer fixture"""
    yield BartSummarizationModel()


@pytest.fixture
def sbert_embedder():
    """SBert embedder fixture"""
    yield SBertEmbeddingModel()


@pytest.fixture
def memory_storage():
    yield MemoryStorage()


@pytest.fixture
def tree_builder(bart_summarization_model, sbert_embedder, memory_storage):
    yield ClusterTreeBuilder(
        ClusterTreeBuilder.Config(
            summarization_model=bart_summarization_model,
            embedding_model=sbert_embedder,
            storage=memory_storage,
        )
    )


@pytest.fixture
def tree_retriever(sbert_embedder, memory_storage):
    yield TreeRetriever(
        TreeRetriever.Config(
            storage=memory_storage,
            limit=TreeRetriever.Limit.TopK(5),
            embedding_model=sbert_embedder,
        )
    )


@pytest.fixture
def cinderella_text():
    with open("tests/cinderella.txt", "r") as f:
        return f.read()


class TreeEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, set):
            return {
                "_raptor_type": "set",
                "value": list(obj),
            }
        if isinstance(obj, np.ndarray):
            return {
                "_raptor_type": "np.ndarray",
                "value": obj.tolist(),
            }
        return json.JSONEncoder.default(self, obj)


class TreeDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if not "_raptor_type" in dct:
            return dct
        assert "value" in dct, "Expected key 'value' with '_raptor_type'"

        if dct["_raptor_type"] == "set":
            return set(dct["value"])
        elif dct["_raptor_type"] == "np.ndarray":
            return np.array(np.array(dct["value"]))
        else:
            raise ValueError(f"Unknown _raptor_type: {dct["_raptor_type"]}")


@pytest.fixture
def save_tree():
    def inner(path: str, storage: MemoryStorage):
        with open(path, "w") as f:
            f.write(json.dumps(storage.to_dict(), cls=TreeEncoder))

    return inner


@pytest.fixture
def cinderella_tree():
    with open("tests/cinderella.json", "r") as f:
        return json.loads(f.read(), cls=TreeDecoder)
