"""CLI entry point for running experiments.

Usage:
    python run_model.py --model phi4 --temp 1.0 [--condition A_nl_consumer]
    python run_model.py --model phi4 --temp 0.0
    python run_model.py --model mistral-small3.2 --temp 1.0 --condition D_general_psych
"""

import argparse
import pathlib
import sys

import yaml
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm

# Add project paths
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "02_render"))
sys.path.insert(0, str(ROOT / "scripts" / "03_runners"))
sys.path.insert(0, str(ROOT / "scripts" / "04_parse"))

from render_prompts import render_prompt
from ollama_runner import OllamaRunner
from parse_responses import parse_response

FORMS_ALL = ["AB_Afirst", "AB_Bfirst", "REP_Afirst", "REP_Bfirst", "CMP_AoverB", "CMP_BoverA"]
FORMS_ABLATION = ["AB_Afirst", "AB_Bfirst"]  # Ablation uses only 2 forms
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]


def load_model_config(model_name: str) -> dict:
    config_path = ROOT / "config" / "models" / f"{model_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)["model"]


def load_experiment_config() -> dict:
    with open(ROOT / "config" / "experiment.yaml") as f:
        return yaml.safe_load(f)["experiment"]


def main():
    parser = argparse.ArgumentParser(description="Run LLM experiment")
    parser.add_argument("--model", required=True, help="Model config name (e.g. phi4)")
    parser.add_argument("--temp", type=float, required=True, help="Temperature (1.0 or 0.0)")
    parser.add_argument("--condition", default="A_nl_consumer",
                        choices=["A_nl_consumer", "B_structured", "C_coded", "D_general_psych"],
                        help="Prompt condition")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks (0=all)")
    args = parser.parse_args()

    # Load configs
    model_cfg = load_model_config(args.model)
    exp_cfg = load_experiment_config()
    env = dotenv_values(ROOT / ".env")

    api_url = env[model_cfg["api_url_env"]]
    reps = exp_cfg["n_replications"]

    # Determine forms
    forms = FORMS_ABLATION if args.condition != "A_nl_consumer" else FORMS_ALL

    # Output directory
    condition_suffix = f"_{args.condition}" if args.condition != "A_nl_consumer" else ""
    temp_suffix = f"_t{args.temp:.1f}"
    out_dir = ROOT / "data" / "raw" / f"{model_cfg['name']}{condition_suffix}{temp_suffix}"

    # Load data
    personas = pd.read_csv(ROOT / "data" / "design" / "personas_384.csv")
    assignment = pd.read_csv(ROOT / "data" / "stimuli" / "assignment.csv")

    # Init runner (per-model num_predict overrides experiment default)
    num_predict = model_cfg.get("num_predict", exp_cfg["max_tokens"])
    runner = OllamaRunner(
        model_name=model_cfg["name"],
        ollama_model=model_cfg["ollama_model"],
        api_url=api_url,
        output_dir=out_dir,
        max_retries=exp_cfg["max_retries"],
        seed=exp_cfg["seed"],
        num_predict=num_predict,
        think=model_cfg.get("think", False),
    )

    # Build task list
    tasks = []
    for _, persona in personas.iterrows():
        profile = persona.to_dict()
        for scenario_id in SCENARIOS:
            # Get assigned version
            asgn = assignment[
                (assignment["ProfileID"] == profile["ProfileID"]) &
                (assignment["ScenarioID"] == scenario_id)
            ]
            version = asgn["Version"].iloc[0]

            for form_id in forms:
                for rep in range(1, reps + 1):
                    task_id = (
                        f"{model_cfg['name']}:{args.condition}:{args.temp}:"
                        f"{profile['ProfileID']}:{scenario_id}:{form_id}:{rep}"
                    )
                    tasks.append({
                        "task_id": task_id,
                        "profile": profile,
                        "scenario_id": scenario_id,
                        "version": version,
                        "form_id": form_id,
                        "rep": rep,
                    })

    if args.limit > 0:
        tasks = tasks[:args.limit]

    # Count already done
    skip = sum(1 for t in tasks if runner.is_done(t["task_id"]))
    remaining = len(tasks) - skip
    print(f"Total tasks: {len(tasks)}, already done: {skip}, remaining: {remaining}")
    print(f"Model: {model_cfg['ollama_model']} @ {api_url}")
    print(f"Temperature: {args.temp}, Condition: {args.condition}")
    print(f"Output: {out_dir}")

    if remaining == 0:
        print("All tasks already completed!")
        return

    # Run
    errors = 0
    with tqdm(total=remaining, desc=f"{model_cfg['name']} t={args.temp}") as pbar:
        for task in tasks:
            if runner.is_done(task["task_id"]):
                continue

            system, user = render_prompt(
                profile=task["profile"],
                scenario_id=task["scenario_id"],
                version=task["version"],
                form_id=task["form_id"],
                replication=task["rep"],
                condition=args.condition,
            )

            result = runner.run_task(
                task_id=task["task_id"],
                system=system,
                prompt=user,
                temperature=args.temp,
                extra_metadata={
                    "profile_id": task["profile"]["ProfileID"],
                    "scenario_id": task["scenario_id"],
                    "version": task["version"],
                    "form_id": task["form_id"],
                    "replication": task["rep"],
                    "condition": args.condition,
                },
            )

            if result and result.get("status") == "error":
                errors += 1

            pbar.update(1)
            pbar.set_postfix(errors=errors)

    print(f"\nDone. Errors: {errors}/{remaining}")


if __name__ == "__main__":
    main()
