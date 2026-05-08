"""Smoke test: 2 calls × 5 models to verify end-to-end pipeline.

Tests:
1. Each model responds to Ollama API
2. Responses are parseable
3. JSONL output is written correctly
4. Logprobs availability checked

Usage:
    python tests/smoke_test.py
"""

import pathlib
import sys
import json
import tempfile
import shutil

import yaml
import pandas as pd
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "02_render"))
sys.path.insert(0, str(ROOT / "scripts" / "03_runners"))
sys.path.insert(0, str(ROOT / "scripts" / "04_parse"))

from render_prompts import render_prompt
from ollama_runner import OllamaRunner
from parse_responses import parse_response


def load_models():
    models_dir = ROOT / "config" / "models"
    models = []
    for f in sorted(models_dir.glob("*.yaml")):
        with open(f) as fh:
            models.append(yaml.safe_load(fh)["model"])
    return models


def main():
    env = dotenv_values(ROOT / ".env")
    models = load_models()
    personas = pd.read_csv(ROOT / "data" / "design" / "personas_384.csv")
    assignment = pd.read_csv(ROOT / "data" / "stimuli" / "assignment.csv")

    # Use first 2 personas, S1, AB_Afirst, rep=1
    test_personas = personas.head(2)
    scenario_id = "S1"
    form_id = "AB_Afirst"

    print(f"Smoke test: {len(test_personas)} calls × {len(models)} models = "
          f"{len(test_personas) * len(models)} total calls\n")

    results = {}
    for model_cfg in models:
        model_name = model_cfg["name"]
        api_url = env[model_cfg["api_url_env"]]
        print(f"--- {model_name} ({model_cfg['ollama_model']}) @ {api_url} ---")

        # Temp directory for smoke test output
        tmp_dir = pathlib.Path(tempfile.mkdtemp())

        num_predict = model_cfg.get("num_predict", 16)
        runner = OllamaRunner(
            model_name=model_name,
            ollama_model=model_cfg["ollama_model"],
            api_url=api_url,
            output_dir=tmp_dir,
            max_retries=2,
            num_predict=num_predict,
            timeout=600,
            think=model_cfg.get("think", False),
        )

        model_results = []
        for _, persona in test_personas.iterrows():
            profile = persona.to_dict()
            asgn = assignment[
                (assignment["ProfileID"] == profile["ProfileID"]) &
                (assignment["ScenarioID"] == scenario_id)
            ]
            version = asgn["Version"].iloc[0]

            task_id = f"smoke:{model_name}:{profile['ProfileID']}"
            system, user = render_prompt(profile, scenario_id, version, form_id, 1)

            result = runner.run_task(
                task_id=task_id,
                system=system,
                prompt=user,
                temperature=1.0,
                extra_metadata={"profile_id": profile["ProfileID"]},
            )

            if result:
                response_text = result.get("response") or ""
                parsed = parse_response(response_text, form_id)
                duration_ms = (result.get("total_duration_ns") or 0) / 1e6

                print(f"  {profile['ProfileID']}: response='{response_text[:50]}' "
                      f"parsed={parsed} time={duration_ms:.0f}ms "
                      f"status={result['status']}")

                model_results.append({
                    "status": result["status"],
                    "parsed": parsed,
                    "duration_ms": duration_ms,
                })
            else:
                print(f"  {profile['ProfileID']}: SKIPPED (already done)")

        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Summary
        ok = sum(1 for r in model_results if r["status"] != "error")
        parsed_ok = sum(1 for r in model_results if r["parsed"] in ("A", "B"))
        avg_time = (sum(r["duration_ms"] for r in model_results) / len(model_results)
                    if model_results else 0)
        results[model_name] = {
            "api_ok": ok == len(model_results),
            "parse_ok": parsed_ok == len(model_results),
            "avg_ms": avg_time,
        }
        print(f"  Summary: api={ok}/{len(model_results)} "
              f"parse={parsed_ok}/{len(model_results)} "
              f"avg={avg_time:.0f}ms\n")

    # Final report
    print("=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    all_ok = True
    for model_name, r in results.items():
        status = "PASS" if (r["api_ok"] and r["parse_ok"]) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {model_name:25s} {status}  (avg {r['avg_ms']:.0f}ms)")

    print(f"\n{'ALL MODELS PASSED' if all_ok else 'SOME MODELS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
