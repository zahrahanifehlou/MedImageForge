"""Active learning: closing the loop Data -> Model -> Errors -> Better data.

Why this module exists:
    Annotation is the scarce resource in medical imaging. A radiologist's hour
    is finite, so the question is not "label more" but "label WHICH cases".
    Active learning's answer: the ones the current model is most unsure about,
    because a case the model already classifies confidently teaches it little.

How we can test that on a fully labelled dataset:
    CT-ICH is already annotated, so we SIMULATE the workflow. Part of the
    training pool is treated as labelled; the rest is an "unlabelled" pool
    whose labels we refuse to look at until a case is selected for review.
    Revealing a selected case's existing label is the simulated radiologist —
    a perfect, instant oracle, which is the one unrealistic part of this
    setup and is stated as such in the report.

The control that makes the experiment meaningful:
    Adding data almost always helps, so "model improved after annotation" is
    not evidence for active learning. The claim under test is narrower:

        uncertainty sampling beats RANDOM sampling at the SAME budget.

    So every round trains three models: the seed model, seed + uncertainty
    selected cases, and seed + randomly selected cases. Without that random
    arm the experiment cannot distinguish a good selection strategy from the
    mere effect of more data.

Why several seeds:
    Step 11 measured a patient-level AUROC confidence interval of roughly
    +/-0.14 on this test split, and showed one patient was worth 0.093 AUROC.
    A single before/after comparison is therefore indistinguishable from
    noise. We repeat the whole experiment over several seeds and report the
    spread, not one flattering number.

Why selection is per PATIENT:
    A radiologist annotates a study, not an isolated slice. Selecting whole
    patients also keeps the uncertainty estimate honest: if some slices of a
    patient were already in training, the model has effectively seen that
    head and its uncertainty on the remaining slices is not representative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from medimageforge.data import Normalization, SliceDataset, load_pseudonym_map, resolve_paths
from medimageforge.logging_utils import get_logger
from medimageforge.metrics import auroc

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------

def entropy(probabilities: np.ndarray) -> np.ndarray:
    """Binary predictive entropy, in bits. Maximal (1.0) at p = 0.5.

    For a binary problem, entropy, least-confidence (1 - max(p, 1-p)) and
    margin sampling are all monotone functions of |p - 0.5|, so they induce
    the SAME ranking. Entropy is used because it generalizes to more classes
    without changing the code's meaning.
    """
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def patient_uncertainty(frame: pd.DataFrame, rule: str = "mean") -> pd.DataFrame:
    """Aggregate slice uncertainty into one score per patient (per study).

    `mean` asks "how unsure is the model about this study overall" — it favours
    studies that are broadly ambiguous. `max` favours a study containing a
    single maximally confusing slice. We default to mean because annotating a
    study costs the radiologist the whole study, so its average
    informativeness is the relevant quantity.
    """
    work = frame.copy()
    work["uncertainty"] = entropy(work["score"].to_numpy())
    grouped = work.groupby("patient")["uncertainty"]
    if rule == "mean":
        scores = grouped.mean()
    elif rule == "max":
        scores = grouped.max()
    else:
        raise ValueError(f"unknown uncertainty rule: {rule!r}")
    return (
        scores.reset_index()
        .rename(columns={"uncertainty": "uncertainty"})
        .sort_values(["uncertainty", "patient"], ascending=[False, True])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# pools and selection
# ---------------------------------------------------------------------------

def stratified_pool_split(
    patients: dict[str, bool], seed_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split train patients into an initially-labelled seed pool and the rest.

    `patients` maps patient -> has_hemorrhage. The seed pool is stratified on
    that flag: a round-0 model trained on an accidentally all-negative pool
    would be useless, and its uncertainty ranking meaningless.
    """
    rng = np.random.default_rng(seed)
    labelled: list[str] = []
    unlabelled: list[str] = []
    for flag in (True, False):
        group = sorted(p for p, positive in patients.items() if positive == flag)
        shuffled = [group[i] for i in rng.permutation(len(group))]
        cut = round(len(shuffled) * seed_fraction)
        labelled.extend(shuffled[:cut])
        unlabelled.extend(shuffled[cut:])
    return sorted(labelled), sorted(unlabelled)


def select_patients(
    candidates: list[str],
    budget: int,
    strategy: str,
    seed: int,
    uncertainty: pd.DataFrame | None = None,
) -> list[str]:
    """Choose `budget` patients to send for annotation.

    strategy 'uncertainty' — the most uncertain studies first.
    strategy 'random'      — the control arm. Same budget, no intelligence.
    """
    if strategy == "random":
        rng = np.random.default_rng(seed)
        pool = sorted(candidates)
        picked = rng.permutation(len(pool))[:budget]
        return sorted(pool[i] for i in picked)

    if strategy == "uncertainty":
        if uncertainty is None:
            raise ValueError("uncertainty strategy requires an uncertainty frame")
        ranked = uncertainty[uncertainty["patient"].isin(candidates)]
        return sorted(ranked.head(budget)["patient"].tolist())

    raise ValueError(f"unknown selection strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# scoring the unlabelled pool with a trained model
# ---------------------------------------------------------------------------

def score_patients(
    run_dir: Path,
    index: pd.DataFrame,
    db_path: Path,
    curated_dir: Path,
    patients: list[str],
) -> pd.DataFrame:
    """Run a saved model over the unlabelled pool and return per-slice scores.

    Loads the weights AND the normalization statistics from the run's own
    record. Re-deriving normalization here would silently score the pool with
    different preprocessing than the model was trained with — the model would
    look more uncertain than it is, and the whole ranking would be garbage.
    """
    import torch

    from medimageforge.model import SliceCNN

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    normalization = Normalization(
        mean=record["environment"]["normalization_mean"],
        std=record["environment"]["normalization_std"],
    )
    image_size = record["config"]["image_size"]

    subset = index[index["patient"].isin(patients)].reset_index(drop=True)
    if subset.empty:
        return pd.DataFrame(columns=["patient", "slice_no", "score"])

    dataset = SliceDataset(
        paths=resolve_paths(subset, load_pseudonym_map(db_path), curated_dir),
        labels=subset["hemorrhage"].to_numpy(),
        patients=subset["patient"].tolist(),
        slice_numbers=subset["slice_no"].tolist(),
        image_size=image_size,
        normalization=normalization,
        augment=False,
    )

    model = SliceCNN(dropout=record["config"]["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scores = []
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            batch = np.stack([dataset[i][0] for i in range(start, min(start + 64, len(dataset)))])
            logits = model(torch.from_numpy(batch).float())
            scores.append(torch.sigmoid(logits).numpy())

    return pd.DataFrame(
        {
            "patient": dataset.patients,
            "slice_no": dataset.slice_numbers,
            "score": np.concatenate(scores),
        }
    )


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------

def write_experiment_index(out_dir: Path, index: pd.DataFrame) -> Path:
    """Write a minimal release index for an experiment arm.

    Deliberately NOT a full published release: experiment arms are scratch
    data, and immutable versioned releases in `datasets/` should mean
    something. Only the adopted result is published as v1.1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.csv"
    index.to_csv(path, index=False)
    return path


def restrict_train_patients(index: pd.DataFrame, labelled: set[str]) -> pd.DataFrame:
    """Keep all of validation/test; keep only labelled patients in train.

    Validation and test are untouched so every arm is compared on exactly the
    same data — the whole point of the experiment.
    """
    keep_train = (index["split"] == "train") & index["patient"].isin(labelled)
    return index[keep_train | (index["split"] != "train")].reset_index(drop=True)


def arm_result(record, arm: str, seed: int, selected: list[str], n_patients: int) -> dict:
    """Pull the comparable numbers out of a finished training run."""
    test = record.metrics["test"]
    return {
        "seed": seed,
        "arm": arm,
        "n_train_patients": n_patients,
        "n_train_slices": record.split_sizes["train"],
        "run_id": record.run_id,
        "test_auroc": round(test["auroc"], 4),
        "test_auroc_ci": test["auroc_ci"],
        "test_recall": round(test["recall"], 4),
        "test_precision": round(test["precision"], 4),
        "selected_patients": selected,
    }


def run_experiment(
    base_release: Path,
    db_path: Path,
    curated_dir: Path,
    work_dir: Path,
    training_config,
    seed_fraction: float = 0.4,
    budget: int = 12,
    seeds: tuple[int, ...] = (0, 1, 2),
    uncertainty_rule: str = "mean",
) -> dict:
    """Run the full seed / uncertainty / random experiment over several seeds.

    Per seed:
      1. split train patients into a labelled seed pool and an unlabelled pool
      2. train the round-0 'seed' model on the seed pool
      3. score the unlabelled pool with it and rank patients by uncertainty
      4. select `budget` patients by uncertainty, and `budget` at random
      5. train one model per selection, on identical validation/test data
    """
    from dataclasses import replace

    from medimageforge.train import run_training

    index = pd.read_csv(base_release / "index.csv")
    brain = index[index["window"] == "brain"]
    train_brain = brain[brain["split"] == "train"]
    has_hemorrhage = (
        train_brain.groupby("patient")["hemorrhage"].max().astype(bool).to_dict()
    )

    results: list[dict] = []
    details: list[dict] = []

    for seed in seeds:
        labelled, unlabelled = stratified_pool_split(has_hemorrhage, seed_fraction, seed)
        log.info(
            "seed %d: %d labelled / %d unlabelled train patients",
            seed, len(labelled), len(unlabelled),
        )

        # --- round 0 -----------------------------------------------------
        seed_dir = work_dir / f"seed{seed}" / "arm-seed"
        write_experiment_index(seed_dir, restrict_train_patients(index, set(labelled)))
        seed_record = run_training(
            seed_dir, db_path, curated_dir, work_dir / f"seed{seed}" / "runs",
            replace(training_config, seed=training_config.seed + seed),
        )
        results.append(arm_result(seed_record, "seed", seed, [], len(labelled)))

        # --- rank the unlabelled pool with the round-0 model --------------
        scores = score_patients(
            work_dir / f"seed{seed}" / "runs" / seed_record.run_id,
            brain, db_path, curated_dir, unlabelled,
        )
        ranking = patient_uncertainty(scores, rule=uncertainty_rule)

        selections = {
            "uncertainty": select_patients(
                unlabelled, budget, "uncertainty", seed, ranking
            ),
            "random": select_patients(unlabelled, budget, "random", seed),
        }
        details.append(
            {
                "seed": seed,
                "labelled_pool": labelled,
                "unlabelled_pool": unlabelled,
                "uncertainty_ranking": ranking.head(budget).to_dict(orient="records"),
                "selections": selections,
                "overlap_between_strategies": sorted(
                    set(selections["uncertainty"]) & set(selections["random"])
                ),
                "positive_patients_selected": {
                    strategy: int(sum(has_hemorrhage[p] for p in picked))
                    for strategy, picked in selections.items()
                },
            }
        )

        # --- the two selection arms --------------------------------------
        for arm, picked in selections.items():
            expanded = set(labelled) | set(picked)
            arm_dir = work_dir / f"seed{seed}" / f"arm-{arm}"
            write_experiment_index(arm_dir, restrict_train_patients(index, expanded))
            record = run_training(
                arm_dir, db_path, curated_dir, work_dir / f"seed{seed}" / "runs",
                replace(training_config, seed=training_config.seed + seed),
            )
            results.append(arm_result(record, arm, seed, picked, len(expanded)))

    from datetime import datetime, timezone

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_version": base_release.name,
            "seed_fraction": seed_fraction,
            "budget": budget,
            "seeds": list(seeds),
            "uncertainty_rule": uncertainty_rule,
            "training": training_config.__dict__.copy(),
        },
        "results": results,
        "details": details,
        "summary": summarize(results),
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate arms across seeds — the honest view of a noisy experiment."""
    frame = pd.DataFrame(results)
    summary = {}
    for arm, group in frame.groupby("arm"):
        summary[arm] = {
            "n_seeds": int(len(group)),
            "mean_test_auroc": round(float(group["test_auroc"].mean()), 4),
            "std_test_auroc": round(float(group["test_auroc"].std(ddof=1)), 4)
            if len(group) > 1
            else None,
            "min_test_auroc": round(float(group["test_auroc"].min()), 4),
            "max_test_auroc": round(float(group["test_auroc"].max()), 4),
            "mean_train_patients": round(float(group["n_train_patients"].mean()), 1),
        }

    # Paired comparisons. Pairing BY SEED matters: seed-to-seed variation is
    # larger than the effect we are looking for, so comparing arm means
    # without pairing would drown the signal in between-seed noise.
    pivot = frame.pivot_table(index="seed", columns="arm", values="test_auroc")
    for name, (better, worse) in {
        "uncertainty_vs_random": ("uncertainty", "random"),
        "uncertainty_vs_seed": ("uncertainty", "seed"),
        "random_vs_seed": ("random", "seed"),
    }.items():
        if {better, worse} <= set(pivot.columns):
            deltas = (pivot[better] - pivot[worse]).dropna()
            summary[name] = {
                "paired_deltas": [round(float(d), 4) for d in deltas],
                **paired_statistics(deltas.to_numpy()),
                "wins": int((deltas > 0).sum()),
                "losses": int((deltas < 0).sum()),
                "ties": int((deltas == 0).sum()),
            }
    return summary


def paired_statistics(deltas: np.ndarray) -> dict:
    """Mean paired difference with a confidence interval and detectable effect.

    Why a CI and not just a mean: a mean difference of +0.045 over 3 seeds and
    -0.006 over 8 seeds are the *same* underlying result seen through
    different amounts of noise. Reporting only the mean invites stopping as
    soon as the number looks good, which is how a coin flip becomes a finding.

    `minimum_detectable_effect` states what this experiment could have found.
    If it exceeds any plausible effect size, a null result says nothing about
    the method — only that the experiment lacked power.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    if n < 2:
        return {
            "n_pairs": n,
            "mean_delta": round(float(deltas.mean()), 4) if n else None,
            "ci_low": None,
            "ci_high": None,
            "significant": False,
            "minimum_detectable_effect": None,
        }

    mean = float(deltas.mean())
    standard_error = float(deltas.std(ddof=1) / np.sqrt(n))
    # Two-sided 95% t critical values for small samples, df = n - 1.
    t_critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                  7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    margin = t_critical * standard_error
    return {
        "n_pairs": n,
        "mean_delta": round(mean, 4),
        "std_delta": round(float(deltas.std(ddof=1)), 4),
        "ci_low": round(mean - margin, 4),
        "ci_high": round(mean + margin, 4),
        # A difference is only claimed when the interval excludes zero.
        "significant": bool(abs(mean) > margin),
        "minimum_detectable_effect": round(2.9 * standard_error, 4),
    }


def render_markdown(report: dict) -> str:
    """The before/after document the roadmap asks for."""
    config = report["config"]
    lines = [
        "# Step 12 — Active learning loop",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Setup",
        "",
        f"- base release: `{config['base_version']}` (validation and test identical in every arm)",
        f"- seed pool: {config['seed_fraction']:.0%} of train patients, stratified by hemorrhage",
        f"- annotation budget: **{config['budget']} patients** per round",
        f"- uncertainty: binary predictive entropy, aggregated per patient by `{config['uncertainty_rule']}`",
        f"- seeds: {config['seeds']}",
        "",
        "The labels already exist, so a selected case is 'annotated' by revealing its",
        "existing label. That simulated reviewer is a perfect, instant oracle — the one",
        "unrealistic element of this experiment.",
        "",
        "## Arms",
        "",
        "| arm | meaning |",
        "|---|---|",
        "| `seed` | round 0: trained on the initial labelled pool only |",
        "| `uncertainty` | seed pool + the most uncertain patients |",
        "| `random` | seed pool + randomly chosen patients (**the control**) |",
        "",
        "Without the random arm, any improvement could simply be the effect of more",
        "data. The claim under test is that uncertainty beats random at equal budget.",
        "",
        "## Results per seed",
        "",
        "| seed | arm | train patients | train slices | test AUROC | 95% CI | recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["results"]:
        ci = row["test_auroc_ci"]
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['n_train_patients']} | "
            f"{row['n_train_slices']} | {row['test_auroc']:.4f} | "
            f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] | {row['test_recall']:.3f} |"
        )

    lines += ["", "## Summary across seeds", "",
              "| arm | seeds | mean test AUROC | std | min | max | mean train patients |",
              "|---|---|---|---|---|---|---|"]
    for arm in ("seed", "uncertainty", "random"):
        if arm not in report["summary"]:
            continue
        s = report["summary"][arm]
        std = "n/a" if s["std_test_auroc"] is None else f"{s['std_test_auroc']:.4f}"
        lines.append(
            f"| {arm} | {s['n_seeds']} | {s['mean_test_auroc']:.4f} | {std} | "
            f"{s['min_test_auroc']:.4f} | {s['max_test_auroc']:.4f} | "
            f"{s['mean_train_patients']} |"
        )

    lines += ["", "## Paired comparisons (pairing by seed)", "",
              "| comparison | n | mean delta | 95% CI | wins/losses | significant? |",
              "|---|---|---|---|---|---|"]
    for name in ("uncertainty_vs_random", "uncertainty_vs_seed", "random_vs_seed"):
        block = report["summary"].get(name)
        if not block or block.get("ci_low") is None:
            continue
        lines.append(
            f"| {name.replace('_', ' ')} | {block['n_pairs']} | "
            f"{block['mean_delta']:+.4f} | [{block['ci_low']:+.4f}, {block['ci_high']:+.4f}] | "
            f"{block['wins']}/{block['losses']} | "
            f"{'**yes**' if block['significant'] else 'no'} |"
        )

    comparison = report["summary"].get("uncertainty_vs_random")
    if comparison and comparison.get("ci_low") is not None:
        lines += [
            "",
            "### Interpretation",
            "",
            f"- paired deltas: {comparison['paired_deltas']}",
            f"- this experiment could only have detected an effect of about "
            f"**{comparison['minimum_detectable_effect']:.3f}** AUROC or larger.",
        ]
        if not comparison["significant"]:
            lines += [
                f"- the 95% interval [{comparison['ci_low']:+.4f}, "
                f"{comparison['ci_high']:+.4f}] includes zero, so this experiment "
                "**cannot distinguish uncertainty sampling from random selection** "
                "on this dataset. That is a statement about statistical power, not "
                "proof that active learning does not work.",
            ]
    lines.append("")
    return "\n".join(lines)
