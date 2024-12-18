import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


class HFSummarizationModel:

    def __init__(self, model_name: str, max_characters: int = 150):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            # else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
        )
        logger.info(
            "Loaded model %s from hugging face into %s",
            model_name,
            self.device,
        )

        self._max_characters = max_characters

    def summarize(self, context) -> str:
        model_inputs = self._tokenizer([context], return_tensors="pt").to(self.device)
        generated_ids = self._model.generate(
            **model_inputs, max_new_tokens=self._max_characters, do_sample=True
        )
        return self._tokenizer.batch_decode(generated_ids)[0].split("[/INST]")[1]
