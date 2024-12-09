from typing import TYPE_CHECKING
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import IEmbeddingModel


class SBertEmbeddingModel:

    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self._model_name = model_name
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, chunk: str):
        embeddings = self.model.encode(chunk)
        assert isinstance(embeddings, np.ndarray)
        return embeddings

    @property
    def slug(self) -> str:
        return f"SBERT/{self._model_name}"


if TYPE_CHECKING:
    _: IEmbeddingModel[str] = SBertEmbeddingModel()
