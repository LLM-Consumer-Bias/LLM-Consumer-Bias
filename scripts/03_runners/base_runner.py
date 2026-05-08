"""Base runner with resume logic, retry, and JSONL output.

Each call is identified by a unique task_id:
  {model}:{condition}:{temp}:{profile_id}:{scenario_id}:{form_id}:{rep}

Results are appended to data/raw/{model_name}/responses.jsonl.
Already-completed task_ids are loaded on startup for resume support.
"""

import json
import pathlib
import time
from datetime import datetime, timezone
from typing import Optional

import jsonlines


class BaseRunner:
    def __init__(
        self,
        model_name: str,
        output_dir: pathlib.Path,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "responses.jsonl"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Load completed task_ids for resume
        self.completed: set = set()
        if self.output_file.exists():
            with jsonlines.open(self.output_file, mode="r") as reader:
                for obj in reader:
                    if obj.get("status") in ("valid", "invalid"):
                        self.completed.add(obj["task_id"])
            print(f"Resuming: {len(self.completed)} tasks already completed")

    def is_done(self, task_id: str) -> bool:
        return task_id in self.completed

    def call_model(self, system: str, prompt: str, temperature: float) -> dict:
        """Override in subclass. Must return dict with 'response' and 'metadata'."""
        raise NotImplementedError

    def run_task(
        self,
        task_id: str,
        system: str,
        prompt: str,
        temperature: float,
        extra_metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """Execute a single task with retry logic. Returns result dict or None if skipped."""
        if self.is_done(task_id):
            return None

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.call_model(system, prompt, temperature)
                record = {
                    "task_id": task_id,
                    "model": self.model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "valid",  # parsing determines final status later
                    "response": result["response"],
                    "temperature": temperature,
                    "attempt": attempt,
                    **result.get("metadata", {}),
                    **(extra_metadata or {}),
                }
                self._append(record)
                self.completed.add(task_id)
                return record

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        # All retries failed
        error_record = {
            "task_id": task_id,
            "model": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "response": None,
            "temperature": temperature,
            "attempt": self.max_retries,
            "error": last_error,
            **(extra_metadata or {}),
        }
        self._append(error_record)
        return error_record

    def _append(self, record: dict) -> None:
        with jsonlines.open(self.output_file, mode="a") as writer:
            writer.write(record)
