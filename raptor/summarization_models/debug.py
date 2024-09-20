"""Debug summarizer"""


class DebugSummarizationModel:
    def summarize(self, context, max_characters=150) -> str:
        return context[:max_characters]
