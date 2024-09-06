from transformers import pipeline
from ._base import BaseSummarizationModel


class BartSummarizationModel(BaseSummarizationModel):

    def __init__(self):
        self.summarizer = pipeline("summarization")

    def summarize(self, context, max_tokens=150) -> str:
        result = self.summarizer(
            context,
            min_length=5,
            max_length=max_tokens,
            return_text=True,
        )

        return result[0]["summary_text"]
