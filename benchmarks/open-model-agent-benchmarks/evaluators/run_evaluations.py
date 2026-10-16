#!/usr/bin/env python3
"""Discover and run benchmark evaluators from category subfolders."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALUATOR_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = EVALUATOR_DIR.parent / "results.json"

CATEGORY_PREFIXES = {
    "reasoning": "R",
    "tool-use": "T",
    "tool-call": "T",
    "web-search": "W",
    "coding": "C",
    "writing": "WR",
    "agent-workflow": "A",
}


def normalize_category(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def discover_evaluators():
    """Find Python task evaluators inside evaluators/<category>/."""
    evaluators = {}

    for path in sorted(EVALUATOR_DIR.glob("*/*.py")):
        if path.name == "__init__.py":
            continue

        module_name = (
            f"benchmark_eval_{path.parent.name.replace('-', '_')}_{path.stem}"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)

        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        task_id = getattr(module, "TASK_ID", None)
        evaluate = getattr(module, "evaluate", None)

        # Ignore Python files that aren't task evaluators.
        if not isinstance(task_id, str) or not callable(evaluate):
            continue

        task_id = task_id.strip().upper()
        category = normalize_category(
            getattr(module, "CATEGORY_ID", path.parent.name)
        )

        if not category:
            raise RuntimeError(f"Missing category for {path}")

        if task_id in evaluators:
            raise RuntimeError(f"Duplicate evaluator for {task_id}")

        evaluators[task_id] = {
            "category": category,
            "module": module,
            "path": path,
        }

    return evaluators


def get_task_code(run: dict[str, Any]) -> str | None:
    """Prefer explicit task metadata; support older task IDs as a fallback."""
    metadata = run.get("taskMetadata")

    if isinstance(metadata, dict):
        source_code = metadata.get("sourceTaskCode")
        if isinstance(source_code, str) and source_code.strip():
            return source_code.strip().upper()

    category = normalize_category(run.get("categoryId"))
    prefix = CATEGORY_PREFIXES.get(category or "")
    task_id = run.get("taskId")

    if prefix and isinstance(task_id, str):
        match = re.search(
            r"(?:^|[-_])task[-_]?0*(\d+)$",
            task_id.strip(),
            re.IGNORECASE,
        )
        if match:
            return f"{prefix}-{int(match.group(1)):02d}"

    return None


def evaluate_run(run, evaluator_module):
    """Send one recorded model response to the task-specific evaluator."""
    response = run.get("responseMarkdown", "")
    result = evaluator_module.evaluate(response)

    if not isinstance(result, dict):
        raise RuntimeError("Evaluator must return a dictionary")

    if result.get("status") not in {"pass", "fail", "manual_review"}:
        raise RuntimeError("Evaluator returned an unsupported status")

    recorded = run.get("recordedStatus")
    result["recordedStatus"] = recorded

    result["statusMatchesRecordedStatus"] = (
        result["status"] == recorded
        if result["status"] in {"pass", "fail"}
        and recorded in {"pass", "fail"}
        else None
    )

    return result


def save_evaluation(run, result):
    """Save verification information without overwriting the original status."""
    evaluation = run.setdefault("evaluation", {})

    if not isinstance(evaluation, dict):
        evaluation = {}
        run["evaluation"] = evaluation

    previous_status = evaluation.get("status")
    if (
        previous_status
        and previous_status != "independently_verified"
        and not evaluation.get("previousEvaluationStatus")
    ):
        evaluation["previousEvaluationStatus"] = previous_status

    status = result["status"]

    evaluation["status"] = (
        "independently_verified"
        if status in {"pass", "fail"}
        else "manual_review_required"
    )
    evaluation["verifiedResult"] = (
        status if status in {"pass", "fail"} else None
    )
    evaluation["evaluator"] = result.get("evaluator")
    evaluation["evaluatorVersion"] = result.get("evaluatorVersion")
    evaluation["verificationMethod"] = "deterministic_task_specific_script"
    evaluation["verificationReason"] = result.get("reason")
    evaluation["statusMatchesRecordedStatus"] = (
        result.get("statusMatchesRecordedStatus")
    )
    evaluation["checkedAt"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    # Save task-specific details without needing to register field names here.
    common_fields = {
        "taskId",
        "status",
        "passed",
        "reason",
        "evaluator",
        "evaluatorVersion",
        "recordedStatus",
        "statusMatchesRecordedStatus",
    }

    for key, value in result.items():
        if key not in common_fields:
            evaluation[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the available benchmark task evaluators."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path to results.json",
    )
    parser.add_argument(
        "--task",
        help="Run just one task, such as R-02. Default: all discovered tasks.",
    )
    parser.add_argument(
        "--category",
        help="Run one category, such as reasoning or tool-use.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Save results. Without this flag, only preview.",
    )
    args = parser.parse_args()

    try:
        evaluators = discover_evaluators()
    except Exception as exc:
        print(f"ERROR discovering evaluators: {exc}", file=sys.stderr)
        return 2

    if not evaluators:
        print("No task evaluators were found.", file=sys.stderr)
        return 2

    requested_task = args.task.strip().upper() if args.task else None
    requested_category = normalize_category(args.category)

    if requested_task and requested_task not in evaluators:
        print(
            f"No evaluator for {requested_task}. "
            f"Available tasks: {', '.join(sorted(evaluators))}",
            file=sys.stderr,
        )
        return 2

    selected_tasks = {
        task_id
        for task_id, info in evaluators.items()
        if (requested_task is None or task_id == requested_task)
        and (
            requested_category is None
            or info["category"] == requested_category
        )
    }

    results_path = args.results.expanduser().resolve()

    if not results_path.is_file():
        print(f"Results file not found: {results_path}", file=sys.stderr)
        return 2

    try:
        with results_path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read results JSON: {exc}", file=sys.stderr)
        return 2

    runs = dataset.get("runs") if isinstance(dataset, dict) else None
    if not isinstance(runs, list):
        print("JSON must contain a top-level 'runs' array.", file=sys.stderr)
        return 2

    targets = []

    for run in runs:
        if not isinstance(run, dict):
            continue

        task_code = get_task_code(run)

        if task_code not in selected_tasks:
            continue

        info = evaluators[task_code]
        run_category = normalize_category(run.get("categoryId"))

        if run_category and run_category != info["category"]:
            continue

        try:
            result = evaluate_run(run, info["module"])
        except Exception as exc:
            print(
                f"ERROR evaluating {run.get('id', '<unknown run>')}: {exc}",
                file=sys.stderr,
            )
            return 2

        targets.append((run, result, task_code))

    print(f"Results file: {results_path}")
    print(f"Available evaluators: {', '.join(sorted(evaluators))}")
    print(f"Matching runs: {len(targets)}")
    print(
        "Mode: SAVE CHANGES"
        if args.write
        else "Mode: PREVIEW ONLY (JSON will not be changed)"
    )

    if not targets:
        print("No matching runs found. Nothing was changed.")
        return 1

    counts = {}

    for run, result, task_code in targets:
        tally = counts.setdefault(
            task_code,
            {"pass": 0, "fail": 0, "manual_review": 0},
        )

        status = result["status"]
        tally[status if status in {"pass", "fail"} else "manual_review"] += 1

        print(
            f"\n[{status.upper()}] {run.get('id', '<unknown run>')} | "
            f"{run.get('harnessId', 'unknown harness')} | "
            f"{run.get('modelId', 'unknown model')}"
        )
        print(f"  {result.get('reason', '')}")

        if task_code == "R-02" and result.get("actualPercentage") is not None:
            print(f"  Detected answer: {result['actualPercentage']:g}%")

        if result.get("statusMatchesRecordedStatus") is False:
            print("  WARNING: verified result differs from recordedStatus.")

    print("\nSummary:")
    for task_code, tally in sorted(counts.items()):
        total = sum(tally.values())
        print(
            f"  {task_code}: {total} checked, "
            f"{tally['pass']} pass, {tally['fail']} fail, "
            f"{tally['manual_review']} manual review"
        )

    if not args.write:
        print("\nPreview only. Add --write to save these results.")
        return 0

    for run, result, _ in targets:
        save_evaluation(run, result)

    backup_path = results_path.with_suffix(results_path.suffix + ".bak")
    temp_path = results_path.with_suffix(results_path.suffix + ".tmp")

    try:
        shutil.copy2(results_path, backup_path)

        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(dataset, file, indent=2, ensure_ascii=False)
            file.write("\n")

        # Validate the output JSON before replacing the original.
        with temp_path.open("r", encoding="utf-8") as file:
            json.load(file)

        temp_path.replace(results_path)

    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not save results: {exc}", file=sys.stderr)
        return 2
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"\nUpdated {len(targets)} runs.")
    print(f"Backup saved to: {backup_path}")
    print("Original recordedStatus values were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())