"""Result aggregation and compact case-study reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import resolve_path, save_dataframe, write_text


def build_run_summary_row(
    cfg: object,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_checkpoint: str,
    train_examples: int,
    pseudo_examples: int,
) -> pd.DataFrame:
    """Build the single-row run summary written after each experiment."""
    row = {
        "experiment_name": str(cfg.experiment.name),
        "seed": int(cfg.seed),
        "shot_count": int(cfg.experiment.shot_count),
        "do_normalization": bool(cfg.experiment.do_normalization),
        "do_self_training": bool(cfg.experiment.do_self_training),
        "pseudo_label_threshold": float(cfg.experiment.pseudo_label_threshold),
        "train_examples": int(train_examples),
        "pseudo_examples": int(pseudo_examples),
        "val_precision": float(validation_metrics["precision"]),
        "val_recall": float(validation_metrics["recall"]),
        "val_f1": float(validation_metrics["f1"]),
        "val_accuracy": float(validation_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "best_checkpoint": best_checkpoint,
    }
    return pd.DataFrame([row])


def load_run_summaries(output_root: str, seed: int) -> pd.DataFrame:
    """Load one-row experiment summaries from the outputs root."""
    root = resolve_path(output_root)
    frames: list[pd.DataFrame] = []
    for summary_path in sorted(root.glob(f"*/seed_{seed}/results_summary.csv")):
        frames.append(pd.read_csv(summary_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def count_error_types(error_csv_path: Path) -> dict[str, int]:
    """Count error categories from a sentence-level error CSV."""
    if not error_csv_path.exists():
        return {
            "boundary_errors": 0,
            "label_confusions": 0,
            "noisy_token_issues": 0,
            "oov_rare_issues": 0,
        }

    error_frame = pd.read_csv(error_csv_path)
    if error_frame.empty:
        return {
            "boundary_errors": 0,
            "label_confusions": 0,
            "noisy_token_issues": 0,
            "oov_rare_issues": 0,
        }

    return {
        "boundary_errors": int(error_frame["boundary_error"].fillna(False).astype(bool).sum()),
        "label_confusions": int(error_frame["label_confusion"].fillna(False).astype(bool).sum()),
        "noisy_token_issues": int(error_frame["noisy_token_issue"].fillna(False).astype(bool).sum()),
        "oov_rare_issues": int(error_frame["oov_rare_issue"].fillna(False).astype(bool).sum()),
    }


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as GitHub-flavored Markdown without extra deps."""
    if frame.empty:
        return "_No rows available._"

    display_frame = frame.copy()
    for column in display_frame.columns:
        if pd.api.types.is_float_dtype(display_frame[column]):
            display_frame[column] = display_frame[column].map(lambda value: f"{value:.4f}")

    headers = [str(column) for column in display_frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display_frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display_frame.columns) + " |")
    return "\n".join(lines)


def compute_cost_score(row: pd.Series) -> int:
    """Cheap heuristic cost score for a student-friendly recommendation."""
    return int(row["shot_count"]) + (75 if bool(row["do_self_training"]) else 0) + (5 if bool(row["do_normalization"]) else 0)


def describe_recipe(row: pd.Series | None) -> str:
    """Turn a result row into a short readable recipe string."""
    if row is None:
        return "No result available."
    return (
        f"{row['experiment_name']} "
        f"(shots={int(row['shot_count'])}, normalization={bool(row['do_normalization'])}, "
        f"self_training={bool(row['do_self_training'])}, test_f1={row['test_f1']:.4f})"
    )


def generate_case_study_artifacts(cfg: object) -> dict[str, Any]:
    """Generate comparison CSVs and a compact Markdown summary."""
    summaries = load_run_summaries(cfg.paths.output_root, int(cfg.seed))
    if summaries.empty:
        return {"comparison_rows": 0}

    baseline_rows = summaries[summaries["experiment_name"] == "source_only"]
    baseline_f1 = float(baseline_rows.iloc[0]["test_f1"]) if not baseline_rows.empty else None

    comparison = summaries.copy()
    if baseline_f1 is not None:
        comparison["absolute_gain_f1"] = comparison["test_f1"] - baseline_f1
        comparison["relative_gain_percent"] = comparison["absolute_gain_f1"].map(
            lambda value: 100.0 * value / max(baseline_f1, 1e-8)
        )
    else:
        comparison["absolute_gain_f1"] = pd.NA
        comparison["relative_gain_percent"] = pd.NA

    comparison["cost_score"] = comparison.apply(compute_cost_score, axis=1)
    comparison = comparison.sort_values(by=["experiment_name", "seed"]).reset_index(drop=True)
    save_dataframe(cfg.paths.results_comparison_path, comparison)

    error_rows: list[dict[str, Any]] = []
    baseline_counts = None
    output_root = resolve_path(cfg.paths.output_root)
    for _, summary_row in comparison.iterrows():
        experiment_name = summary_row["experiment_name"]
        error_counts = count_error_types(output_root / experiment_name / f"seed_{int(cfg.seed)}" / "error_analysis_sample.csv")
        if experiment_name == "source_only":
            baseline_counts = error_counts

        row = {"experiment_name": experiment_name, **error_counts}
        if baseline_counts is not None:
            for key, value in error_counts.items():
                base_value = baseline_counts[key]
                reduction = base_value - value
                row[f"{key}_reduction"] = reduction
                row[f"{key}_reduction_percent"] = 100.0 * reduction / max(base_value, 1)
        error_rows.append(row)

    error_reduction = pd.DataFrame(error_rows)
    save_dataframe(cfg.paths.error_type_reduction_path, error_reduction)

    adapted = comparison[comparison["experiment_name"] != "source_only"].copy()
    cheapest_effective = None
    if baseline_f1 is not None and not adapted.empty:
        positive_gain = adapted[adapted["absolute_gain_f1"] > 0].copy()
        if not positive_gain.empty:
            cheapest_effective = positive_gain.sort_values(by=["cost_score", "test_f1"], ascending=[True, False]).iloc[0]

    best_low_label = None
    low_label_pool = adapted[adapted["shot_count"] <= 100].copy()
    if not low_label_pool.empty:
        best_low_label = low_label_pool.sort_values(by="test_f1", ascending=False).iloc[0]

    best_overall = None
    if not adapted.empty:
        best_overall = adapted.sort_values(by="test_f1", ascending=False).iloc[0]

    if best_overall is not None:
        if cheapest_effective is not None and (best_overall["test_f1"] - cheapest_effective["test_f1"]) <= 0.01:
            recommendation = cheapest_effective
        else:
            recommendation = best_overall
    else:
        recommendation = comparison.sort_values(by="test_f1", ascending=False).iloc[0]

    comparison_columns = [
        "experiment_name",
        "shot_count",
        "do_normalization",
        "do_self_training",
        "test_f1",
        "absolute_gain_f1",
        "relative_gain_percent",
    ]
    error_columns = [
        "experiment_name",
        "boundary_errors",
        "label_confusions",
        "noisy_token_issues",
        "oov_rare_issues",
    ]

    summary_lines = [
        "# Case Study Summary",
        "",
        "This summary is generated from local run artifacts for the NER-only domain-adaptation case study.",
        "",
        "## Results comparison",
        "",
        dataframe_to_markdown(comparison[comparison_columns]),
        "",
        "## Error type counts",
        "",
        dataframe_to_markdown(error_reduction[error_columns]),
        "",
        "## Takeaways",
        "",
        f"- Cheapest effective recipe: {describe_recipe(cheapest_effective)}",
        f"- Best low-label recipe: {describe_recipe(best_low_label)}",
        f"- Best overall recipe: {describe_recipe(best_overall)}",
        f"- Practical 4-5 week recommendation: {describe_recipe(recommendation)}",
    ]
    write_text(cfg.paths.case_study_summary_path, "\n".join(summary_lines))
    return {"comparison_rows": int(len(comparison)), "error_rows": int(len(error_reduction))}

