import logging
from abc import ABC, abstractmethod
import numpy as np

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def slug(self) -> str:
        raise NotImplementedError("Implement in subclass")
