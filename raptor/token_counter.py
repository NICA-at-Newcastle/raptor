from abc import abstractmethod, ABC
import math


class BaseTokenCounter(ABC):
    """Base class for all token counters."""

    @abstractmethod
    def __call__(self, text: str) -> int:
        """
        Returns the number of tokens that would be present in the encoded text.
        """
        raise NotImplementedError("Implement in subclass")


class BytePairTokenCounter(BaseTokenCounter):
    """Token counter which assumes that each token = 2 bytes."""

    def __call__(self, text: str) -> int:
        return math.ceil(len(text.encode()) / 2)
