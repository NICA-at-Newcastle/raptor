import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseEmbeddingModel


class SBertEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self._model_name = model_name
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        embeddings = self.model.encode(text)
        assert isinstance(embeddings, np.ndarray)
        return embeddings

    @property
    def slug(self) -> str:
        return f"SBERT/{self._model_name}"
