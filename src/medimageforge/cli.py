"""Command-line entry point: `python -m medimageforge <command>`.

Why this module exists:
    Every pipeline step becomes a subcommand here (`info` today; `explore`,
    `ingest`, `curate`, `qc`, ... in later steps). One entry point means one
    place where config is loaded and logging is configured — the "front door"
    of the platform.
"""

from __future__ import annotations

import argparse

import numpy as np

from medimageforge import __version__
from medimageforge.config import data_path, load_config
from medimageforge.curate import curate, curation_state, write_report
from medimageforge.explore import (
    LABEL_COLUMNS,
    label_slice_exists,
    load_demographics,
    load_labels,
    scan_raw_dir,
    verify_checksums,
)
from medimageforge.imaging import (
    find_masks,
    load_gray,
    mask_from_jpeg,
    overlay_mask,
    pixel_stats,
    side_by_side,
)
from medimageforge.logging_utils import configure_logging, get_logger
from medimageforge.manifest import ingest

log = get_logger(__name__)


def cmd_info(config: dict) -> int:
    """Print the resolved configuration and check which paths exist.

    This is our smoke test: if `info` works, config loading and path
    resolution work, and we can see whether the dataset is in place.
    """
    print(f"medimageforge {__version__}")
    print(f"log level: {config['logging']['level']}")
    print("\nConfigured paths:")
    for key in config["paths"]:
        p = data_path(config, key)
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status:7s}] {key}: {p}")
    return 0


def cmd_explore(config: dict) -> int:
    """Scan the dataset and print a report: what is on disk vs what is claimed.

    The explorer is deliberately read-only — measuring the data changes
    nothing. That is what makes it safe to run anytime.
    """
    raw_dir = data_path(config, "raw_dir")
    data_dir = data_path(config, "data_dir")
    windows = config["dataset"]["windows"]
    mask_suffix = config["dataset"]["mask_suffix"]
    width = config["dataset"]["patient_id_width"]

    patients = scan_raw_dir(raw_dir, windows, mask_suffix)
    labels = load_labels(data_path(config, "labels_csv"))
    demographics = load_demographics(data_path(config, "demographics_csv"))

    print("=== Dataset report ===")
    print(f"Patients on disk:      {len(patients)}")
    for window in windows:
        total = sum(p.slice_counts.get(window, 0) for p in patients)
        print(f"Slices ({window:5s}):      {total}")
    print(f"Segmentation masks:    {sum(p.mask_count for p in patients)}")

    counts = [p.total_slices for p in patients]
    print(f"Slices per patient:    min {min(counts)} / "
          f"mean {sum(counts) / len(counts):.1f} / max {max(counts)}")

    print("\n=== Labels (hemorrhage_diagnosis.csv) ===")
    print(f"Label rows:            {len(labels)}")
    print(f"Patients with labels:  {labels['PatientNumber'].nunique()}")
    for col in LABEL_COLUMNS:
        if col in labels.columns:
            print(f"  {col:20s} {int(labels[col].sum())}")

    print("\n=== Demographics (patient_demographics.csv) ===")
    print(f"Rows:                  {len(demographics)}")
    gender_col = next((c for c in demographics.columns if c.lower() == "gender"), None)
    if gender_col:
        print(f"Gender:                {dict(demographics[gender_col].value_counts())}")

    print("\n=== Labels vs files ===")
    missing_slices = label_slice_exists(labels, raw_dir, width)
    if missing_slices:
        print(f"Labels with no brain slice on disk: {len(missing_slices)}")
        for rel in missing_slices[:10]:
            print(f"  - {rel}")
        if len(missing_slices) > 10:
            print(f"  ... and {len(missing_slices) - 10} more")
    else:
        print("Every label row has a matching brain slice.")

    print("\n=== Integrity (SHA256SUMS.txt) ===")
    log.info("Verifying checksums — hashing every file in data/ ...")
    checks = verify_checksums(data_dir, data_path(config, "checksums_file"))
    print(f"Verified:   {checks.verified}")
    print(f"Mismatched: {len(checks.mismatched)}")
    print(f"Missing:    {len(checks.missing)}")
    print(f"Extra:      {len(checks.extra)}")
    for rel in (checks.mismatched + checks.missing + checks.extra)[:10]:
        print(f"  - {rel}")

    ok = checks.ok and not missing_slices
    print(f"\nIntegrity gate: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_inspect(config: dict, patient: str, slice_no: int | None) -> int:
    """Render one slice: bone window | brain window | brain + mask overlay.

    If --slice is omitted we pick the first slice that has a mask — those
    are the interesting ones. The composite PNG is written to artifacts/
    because this CLI has no display to show() on.
    """
    from PIL import Image

    raw_dir = data_path(config, "raw_dir")
    mask_suffix = config["dataset"]["mask_suffix"]
    brain_dir = raw_dir / patient / "brain"
    if not brain_dir.is_dir():
        print(f"No such patient folder: {patient}")
        return 1

    if slice_no is None:
        masks = find_masks(brain_dir, mask_suffix)
        if not masks:
            print(f"Patient {patient} has no masks; pass --slice explicitly.")
            return 1
        slice_no = int(masks[0].name.split("_")[0])
        print(f"Auto-picked slice {slice_no} (has a segmentation mask)")

    brain_path = brain_dir / f"{slice_no}.jpg"
    bone_path = raw_dir / patient / "bone" / f"{slice_no}.jpg"
    mask_path = brain_dir / f"{slice_no}{mask_suffix}.jpg"
    for p in (brain_path, bone_path):
        if not p.is_file():
            print(f"Missing file: {p.relative_to(data_path(config, 'data_dir'))}")
            return 1

    brain = load_gray(brain_path)
    bone = load_gray(bone_path)
    print(f"brain {brain_path.name}: {pixel_stats(brain)}")
    print(f"bone  {bone_path.name}: {pixel_stats(bone)}")

    if mask_path.is_file():
        mask = mask_from_jpeg(load_gray(mask_path))
        overlay = overlay_mask(brain, mask)
        print(f"mask  {mask_path.name}: {int(mask.sum())} px "
              f"({mask.mean() * 100:.1f}% of image)")
    else:
        overlay = np.stack([brain] * 3, axis=-1)
        print(f"mask  {mask_path.name}: none — overlay is plain brain window")

    composite = side_by_side(bone, brain, overlay)
    out_dir = data_path(config, "artifacts_dir") / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient}_{slice_no}.png"
    Image.fromarray(composite).save(out_path)
    print(f"\nSaved: {out_path}")
    print("Panels: bone window | brain window | brain + mask overlay")
    return 0


def cmd_ingest(config: dict) -> int:
    """Register every file under data/ into the SQLite manifest.

    Re-runnable by design: the second run on unchanged data should report
    all files 'unchanged' — that is how you know the pipeline is idempotent.
    """
    db_path = data_path(config, "manifest_db")
    log.info("Ingesting %s into %s", data_path(config, "data_dir"), db_path)
    report = ingest(
        data_path(config, "data_dir"),
        db_path,
        config["dataset"]["windows"],
        config["dataset"]["mask_suffix"],
    )
    print("=== Ingest report ===")
    print(f"Files on disk:  {report.total_on_disk}")
    print(f"New:            {report.new}")
    print(f"Unchanged:      {report.unchanged}")
    print(f"Updated:        {report.updated}")
    print(f"Missing:        {report.missing}")
    print(f"Manifest:       {db_path}")
    return 0


def cmd_curate(config: dict, force: bool) -> int:
    """Validate registered files and write normalized copies to the curated zone."""
    db_path = data_path(config, "manifest_db")
    if not db_path.is_file():
        print("No manifest found — run `python -m medimageforge ingest` first.")
        return 1

    curated_dir = data_path(config, "curated_dir")
    labels = load_labels(data_path(config, "labels_csv"))
    log.info("Curating into %s", curated_dir)
    report = curate(
        data_dir=data_path(config, "data_dir"),
        curated_dir=curated_dir,
        db_path=db_path,
        labels=labels,
        expected_size=tuple(config["dataset"]["expected_size"]),
        mask_threshold=config["dataset"]["mask_threshold"],
        force=force,
    )

    state = curation_state(db_path)

    print("=== This run ===")
    print(f"Accepted: {report.accepted}")
    print(f"Rejected: {report.rejected}")
    print(f"Skipped (already curated, source unchanged): {report.skipped}")

    # The state view is the honest one: it still reports anomalies after an
    # idempotent re-run, when "this run" did nothing at all.
    print("\n=== Curated zone state ===")
    print(f"Accepted: {state['accepted']}")
    print(f"Rejected: {state['rejected']}")

    if state["rejections"]:
        print("\nRejections (every one, with reason):")
        for item in state["rejections"][:20]:
            print(f"  - {item['rel_path']}: {', '.join(item['reasons'])}")
        if len(state["rejections"]) > 20:
            print(f"  ... and {len(state['rejections']) - 20} more (see JSON report)")
    else:
        print("\nNo rejections — every validated file passed.")

    if state["warnings"]:
        print("\nWarnings (curated, but noteworthy):")
        for item in state["warnings"][:20]:
            print(f"  ! {item['rel_path']}: {', '.join(item['warnings'])}")

    report_path = data_path(config, "artifacts_dir") / "curation_report.json"
    write_report(report, state, report_path)
    print(f"\nCurated zone: {curated_dir}")
    print(f"JSON report:  {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="medimageforge")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (default: configs/default.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Show resolved configuration and path status")
    sub.add_parser("explore", help="Scan the dataset and print an integrity report")
    p_inspect = sub.add_parser(
        "inspect", help="Render a slice: bone | brain | brain+mask overlay"
    )
    p_inspect.add_argument("patient", help="Patient folder name, e.g. 049")
    p_inspect.add_argument("--slice", type=int, default=None, help="Slice number")
    sub.add_parser("ingest", help="Register all data files into the SQLite manifest")
    p_curate = sub.add_parser(
        "curate", help="Validate files and write normalized copies to the curated zone"
    )
    p_curate.add_argument(
        "--force", action="store_true", help="Re-curate files even if already done"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config["logging"]["level"])
    log.debug("Loaded config: %s", config)

    if args.command == "info":
        return cmd_info(config)
    if args.command == "explore":
        return cmd_explore(config)
    if args.command == "inspect":
        return cmd_inspect(config, args.patient, args.slice)
    if args.command == "ingest":
        return cmd_ingest(config)
    if args.command == "curate":
        return cmd_curate(config, args.force)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
