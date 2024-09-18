# pylint: disable=missing-function-docstring, too-few-public-methods

from typing import TypeVar, Generic, Protocol

_CHUNK = TypeVar("_CHUNK")


class ISummarizationModel(Protocol, Generic[_CHUNK]):
    """Defines the methods required for a raptor summarization model"""

    def summarize(self, chunks: list[_CHUNK], max_tokens=150) -> _CHUNK: ...
