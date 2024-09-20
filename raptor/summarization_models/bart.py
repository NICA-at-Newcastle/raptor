from transformers import pipeline


class BartSummarizationModel:

    def __init__(self):
        self.summarizer = pipeline("summarization")

    def summarize(self, context, max_characters=150) -> str:
        result = self.summarizer(
            context,
            min_length=5,
            max_length=max_characters,
            return_text=True,
        )

        return result[0]["summary_text"]
