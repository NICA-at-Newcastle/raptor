"""Summarization models for OpenAI and OpenAI-compatible APIs. """

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


class OpenAiSummarizationModel:
    """
    Works with any Open AI compatible API.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        endpoint: str | None = None,
    ):
        self._client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )
        self._model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_characters=150) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                },
            ],
            max_tokens=max_characters,
            temperature=0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Model failed to generate completion: {response}")

        return content


class GPT3TurboSummarizationModel(OpenAiSummarizationModel):
    """GPT3 Turbo using Open AI's API"""

    def __init__(self):
        super().__init__("gpt-3.5-turbo")


class GPT3SummarizationModel(OpenAiSummarizationModel):
    """GPT3 using Open AI's Davinci API"""

    def __init__(
        self,
    ):
        super().__init__("text-davinci-003")
