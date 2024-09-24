from transformers import pipeline


class BartSummarizationModel:

    def __init__(self, max_characters=150):
        self.summarizer = pipeline("summarization")
        self._max_characters = max_characters

    def summarize(self, context) -> str:
        result = self.summarizer(
            context,
            min_length=5,
            max_length=self._max_characters,
            return_text=True,
        )

        return result[0]["summary_text"]
