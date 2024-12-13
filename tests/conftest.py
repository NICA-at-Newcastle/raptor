import json
import pytest
import numpy as np
from raptor import ClusterTreeBuilder
from raptor.summarization_models import BartSummarizationModel
from raptor.embedding_models import SBertEmbeddingModel
from raptor.storages.memory_storage import MemoryStorage


@pytest.fixture(scope="session")
def memory_storage():
    yield MemoryStorage()


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def build_tree(
    cinderella_text,
    memory_storage,
):

    bart_summarization_model = BartSummarizationModel()
    sbert_embedder = SBertEmbeddingModel()

    builder = ClusterTreeBuilder(
        ClusterTreeBuilder.Config(
            summarization_model=bart_summarization_model,
            embedding_model=sbert_embedder,
            storage=memory_storage,
        )
    )
    chunks = cinderella_text.split(".")
    builder.build_from_chunks(
        chunks,
    )
