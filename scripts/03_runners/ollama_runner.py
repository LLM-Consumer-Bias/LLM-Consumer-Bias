"""Ollama API runner. POST to local Ollama server/api/generate."""

import pathlib
import requests
from typing import Optional

from base_runner import BaseRunner


class OllamaRunner(BaseRunner):
    def __init__(
        self,
        model_name: str,
        ollama_model: str,
        api_url: str,
        output_dir: pathlib.Path,
        max_retries: int = 3,
        seed: int = 20250101,
        num_predict: int = 16,
        timeout: int = 600,
        think: bool = False,
    ):
        super().__init__(model_name, output_dir, max_retries)
        self.ollama_model = ollama_model
        self.api_url = api_url.rstrip("/")
        self.seed = seed
        self.num_predict = num_predict
        self.timeout = timeout
        self.think = think

    def call_model(self, system: str, prompt: str, temperature: float) -> dict:
        """POST to Ollama /api/generate and return response + metadata."""
        payload = {
            "model": self.ollama_model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": temperature,
                "top_p": 1.0,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }

        resp = requests.post(
            f"{self.api_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "response": data.get("response", "").strip(),
            "metadata": {
                "total_duration_ns": data.get("total_duration"),
                "eval_count": data.get("eval_count"),
                "eval_duration_ns": data.get("eval_duration"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
                "done_reason": data.get("done_reason"),
            },
        }
