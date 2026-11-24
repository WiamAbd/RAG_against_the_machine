"""Load Qwen and generate answers using Transformers."""

from collections.abc import Iterable
from typing import Protocol, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    logging,
)


logging.set_verbosity_error()


class TokenizerInterface(Protocol):
    """Describe the tokenizer features used by this wrapper."""

    pad_token_id: int | None
    eos_token_id: int | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        **kwargs: object,
    ) -> BatchEncoding:
        """Apply the chat template and return encoded inputs."""
        ...

    def decode(
        self,
        token_ids: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Convert generated token IDs into text."""
        ...


class ModelConfigInterface(Protocol):
    """Describe the model configuration used by the pipeline."""

    max_position_embeddings: int


class ModelInterface(Protocol):
    """Describe the causal model features used by this project."""

    config: ModelConfigInterface

    def to(self, device: str) -> object:
        """Move the model to a computation device."""
        ...

    def eval(self) -> object:
        """Enable evaluation mode."""
        ...

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return the model parameters."""
        ...

    def generate(self, **kwargs: object) -> torch.Tensor:
        """Generate token IDs."""
        ...


class Small_LLM_Model:
    """Load a causal language model and expose inference helpers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        # Auto-select device with priority: mps > cuda > cpu.
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self._device = device

        if dtype is None:
            dtype = (
                torch.float16
                if self._device in ["cuda", "mps"]
                else torch.float32
            )

        self._dtype = dtype

        self._tokenizer = cast(
            TokenizerInterface,
            AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            ),
        )

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = (
                self._tokenizer.eos_token_id
            )

        self._model = cast(
            ModelInterface,
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self._dtype,
                device_map=(
                    "auto" if self._device == "cuda" else None
                ),
                trust_remote_code=trust_remote_code,
            ),
        )

        self._model.to(self._device)
        self._model.eval()

        for parameter in self._model.parameters():
            parameter.requires_grad = False

    def _prepare_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        """Apply the chat template and return CPU input tensors."""
        messages = [{"role": "user", "content": prompt}]

        encoded = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )

        return {
            name: cast(torch.Tensor, value)
            for name, value in encoded.items()
        }

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate a response using Qwen's chat format."""
        inputs = {
            name: tensor.to(self._device)
            for name, tensor in self._prepare_inputs(prompt).items()
        }

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                return_dict_in_generate=False,
            )

        # Exclude the input prompt from the returned response.
        input_length = inputs["input_ids"].shape[1]
        answer_ids = outputs[0, input_length:]

        return self._tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
        )

    def count_prompt_tokens(self, prompt: str) -> int:
        """Count input tokens, including Qwen's chat template."""
        inputs = self._prepare_inputs(prompt)

        return int(inputs["input_ids"].shape[1])
