"""Generate with vLLM. This module only runs inside the Modal GPU container."""

from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from steering.generation import Completion
from steering.samples import CHOOSE_STRATEGY, CHOOSE_STRATEGY_MAX_NEW_TOKENS, Sample

MAX_MODEL_LENGTH = 4096


class VllmBackend:
    def __init__(self, model_id: str, revision: str, sampling: dict, symbols_dir: Path, seed: int):
        self.sampling = sampling
        self.symbols_dir = symbols_dir
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.llm = LLM(
            model=model_id,
            revision=revision,
            tokenizer_revision=revision,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LENGTH,
            limit_mm_per_prompt={"image": 1},
            seed=seed,
        )

    def generate(self, samples: list[Sample]) -> list[Completion]:
        requests = [self._build_request(sample) for sample in samples]
        sampling_params = [self._sampling_params(sample) for sample in samples]
        outputs = self.llm.generate(requests, sampling_params, use_tqdm=False)
        return [self._to_completion(output) for output in outputs]

    def image_token_count(self, image_name: str) -> int:
        """How many prompt tokens one image adds, measured with the model's processor."""
        image = Image.open(self.symbols_dir / f"{image_name}.png").convert("RGB")
        return self._prompt_token_count(image) - self._prompt_token_count(None)

    def text_token_count(self, text: str) -> int:
        return len(self.processor.tokenizer(f" {text}", add_special_tokens=False)["input_ids"])

    def _prompt_token_count(self, image) -> int:
        content = [{"type": "text", "text": "x"}]
        if image is not None:
            content.insert(0, {"type": "image"})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        images = [image] if image is not None else None
        return len(self.processor(text=[prompt], images=images)["input_ids"][0])

    def _build_request(self, sample: Sample) -> dict:
        """Format the prompt with the model's chat template. There is no system prompt."""
        content = [{"type": "text", "text": sample.prompt_text}]
        if sample.has_image:
            content.insert(0, {"type": "image"})

        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        request = {"prompt": prompt}
        if sample.has_image:
            image_path = self.symbols_dir / f"{sample.condition}.png"
            request["multi_modal_data"] = {"image": Image.open(image_path).convert("RGB")}
        return request

    def _sampling_params(self, sample: Sample) -> SamplingParams:
        settings = {key: value for key, value in self.sampling.items() if key != "max_new_tokens"}
        if sample.task == CHOOSE_STRATEGY:
            max_tokens = CHOOSE_STRATEGY_MAX_NEW_TOKENS
        else:
            max_tokens = self.sampling["max_new_tokens"]
        return SamplingParams(
            **settings,
            max_tokens=max_tokens,
            seed=sample.seed,
        )

    @staticmethod
    def _to_completion(output) -> Completion:
        generated = output.outputs[0]
        return Completion(
            text=generated.text,
            prompt_tokens=len(output.prompt_token_ids),
            output_tokens=len(generated.token_ids),
            finish_reason=generated.finish_reason,
        )
