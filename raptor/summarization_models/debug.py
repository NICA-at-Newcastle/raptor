"""Debug summarizer"""


class DebugSummarizationModel:
    def summarize(self, context, max_tokens=150) -> str:
        return context[:max_tokens]
