from typing import TYPE_CHECKING
from transformers import pipeline

from ._base import ISummarizationModel


class BartSummarizationModel:

    def __init__(self, max_characters=150):
        self.summarizer = pipeline("summarization")
        self._max_characters = max_characters

    def summarize(self, chunks) -> str:
        result = self.summarizer(
            chunks,
            min_length=5,
            max_length=self._max_characters,
            return_text=True,
        )

        return result[0]["summary_text"]


if TYPE_CHECKING:
    _: ISummarizationModel = BartSummarizationModel()
