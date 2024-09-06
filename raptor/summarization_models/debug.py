"""Debug summarizer"""

from ._base import BaseSummarizationModel


class DebugSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150) -> str:
        return context[:max_tokens]
