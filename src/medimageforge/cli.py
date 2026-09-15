"""Command-line entry point: `python -m medimageforge <command>`.

Why this module exists:
    Every pipeline step becomes a subcommand here (`info` today; `explore`,
    `ingest`, `curate`, `qc`, ... in later steps). One entry point means one
    place where config is loaded and logging is configured — the "front door"
    of the platform.
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np

from medimageforge import __version__
from medimageforge.config import data_path, load_config
from medimageforge.annotations import (
    get_slice_annotation,
    label_distribution,
    load_annotations,
)
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
from medimageforge.privacy import (
    PrivacyReport,
    build_pseudonym_map,
    check_age_rule,
    export_deidentified,
    find_id_leaks,
    load_or_create_salt,
    pseudonymize_frames,
    scan_columns,
    scan_free_text,
    scan_image_metadata,
)
from medimageforge.privacy import write_report as write_privacy_report
from medimageforge.qc import run_qc
from medimageforge.qc import write_report as write_qc_report
from medimageforge.release import (
    assert_no_patient_overlap,
    build_index,
    patient_strata,
    split_patients,
    split_statistics,
    verify_release,
    write_release,
)

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


def cmd_privacy(config: dict, sample: int) -> int:
    """Run the privacy gate, then pseudonymize the working artifacts.

    Returns non-zero when the gate fails — that is what makes it a gate and
    not a report: a CI job or a later pipeline step can simply check the
    exit code and refuse to continue.
    """
    db_path = data_path(config, "manifest_db")
    if not db_path.is_file():
        print("No manifest found — run `python -m medimageforge ingest` first.")
        return 1

    raw_dir = data_path(config, "raw_dir")
    labels = load_labels(data_path(config, "labels_csv"))
    demographics = load_demographics(data_path(config, "demographics_csv"))
    report = PrivacyReport()

    # --- 1. PHI must not hide in image metadata --------------------------
    images = sorted(raw_dir.rglob("*.jpg"))
    scanned = images if sample <= 0 else images[:sample]
    findings = scan_image_metadata(scanned)
    report.add(
        "image-metadata",
        not findings,
        f"scanned {len(scanned)} images for EXIF/comment segments",
        findings,
    )

    # --- 2. No column may hold a direct identifier -----------------------
    col_findings = scan_columns(labels, "hemorrhage_diagnosis.csv") + scan_columns(
        demographics, "patient_demographics.csv"
    )
    report.add(
        "identifier-columns",
        not col_findings,
        "checked column names against the forbidden-identifier list",
        col_findings,
    )

    # --- 3. Free text must not contain identifiers -----------------------
    text_findings = scan_free_text(labels, "hemorrhage_diagnosis.csv") + scan_free_text(
        demographics, "patient_demographics.csv"
    )
    report.add(
        "free-text-identifiers",
        not text_findings,
        "regex-scanned every string cell for emails/phones/dates/names",
        text_findings,
    )

    # --- 4. HIPAA Safe Harbor age rule -----------------------------------
    age_findings = check_age_rule(demographics, "patient_demographics.csv")
    report.add(
        "age-over-89",
        not age_findings,
        "ages above 89 must be grouped, not reported exactly",
        age_findings,
    )

    # --- 5. Pseudonymize, then prove the export is clean ------------------
    salt = load_or_create_salt(data_path(config, "salt_file"))
    mapping = build_pseudonym_map(db_path, salt)
    items = pseudonymize_frames(
        labels, demographics, mapping, config["dataset"]["patient_id_width"]
    )
    leak_findings = []
    for item in items:
        for finding in find_id_leaks(item["frame"], item["real_ids"]):
            leak_findings.append({"source": item["name"], **finding})
    report.add(
        "no-real-ids-in-exports",
        not leak_findings,
        f"row-aligned check that no column in {len(items)} exports reproduces a patient ID",
        leak_findings,
    )
    exported = export_deidentified(items, data_path(config, "deid_dir"))

    print("=== Privacy gate ===")
    for check in report.checks:
        print(f"  [{check['result']}] {check['check']}: {check['detail']}")
        for item in check["findings"][:5]:
            print(f"        - {item}")
        if len(check["findings"]) > 5:
            print(f"        ... and {len(check['findings']) - 5} more")

    print(f"\nPatients pseudonymized: {len(mapping)}")
    example = sorted(mapping)[0]
    print(f"  example: {example} -> {mapping[example]}")
    print("De-identified exports:")
    for path in exported:
        print(f"  {path}")

    report_path = data_path(config, "artifacts_dir") / "privacy_report.json"
    write_privacy_report(report, report_path)
    print(f"\nJSON report: {report_path}")
    print(f"\nGATE: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


def cmd_load_labels(config: dict) -> int:
    """Translate the wide labels CSV into the normalized label store."""
    db_path = data_path(config, "manifest_db")
    if not db_path.is_file():
        print("No manifest found — run `python -m medimageforge ingest` first.")
        return 1

    labels_csv = data_path(config, "labels_csv")
    report = load_annotations(
        db_path,
        load_labels(labels_csv),
        labels_csv,
        config["dataset"]["patient_id_width"],
    )
    print("=== Label store loaded ===")
    print(f"Slice annotations: {report.annotations}")
    print(f"Label assertions:  {report.labels}")
    print(f"With mask:         {report.with_mask}")
    print(f"Hemorrhage-positive slices: {report.positive_slices}")

    print("\nLabel distribution (from the store, not the CSV):")
    for row in label_distribution(db_path):
        print(f"  [{row['category']:16s}] {row['display_name']:16s} "
              f"{row['positives']:5d} / {row['total']}")
    return 0


def cmd_show_slice(config: dict, patient: str, slice_no: int) -> int:
    """One call: labels + mask path + provenance for a single slice."""
    db_path = data_path(config, "manifest_db")
    records = get_slice_annotation(db_path, patient, slice_no)
    if not records:
        print(f"No annotation found for patient {patient} slice {slice_no}.")
        return 1

    for record in records:
        print(f"=== Patient {record['patient_id']} slice {record['slice_no']} ===")
        print(f"Hemorrhage types: {record['hemorrhage_types'] or '(none)'}")
        print(f"Other findings:   "
              f"{[c for c in record['positive_labels'] if c not in record['hemorrhage_types']] or '(none)'}")
        print(f"No hemorrhage (derived): {record['no_hemorrhage']}")
        print(f"Mask: {record['mask_rel_path'] or '(none)'}")
        print("Provenance:")
        for key, value in record["provenance"].items():
            shown = value[:16] + "..." if key == "source_sha256" else value
            print(f"  {key}: {shown}")
        print(f"All labels: {record['labels']}")
    return 0


def cmd_qc(config: dict, skip_leakage: bool) -> int:
    """Run the quality gates. Non-zero exit when any ERROR is found."""
    db_path = data_path(config, "manifest_db")
    if not db_path.is_file():
        print("No manifest found — run `python -m medimageforge ingest` first.")
        return 1

    report, leakage = run_qc(
        db_path=db_path,
        data_dir=data_path(config, "data_dir"),
        labels=load_labels(data_path(config, "labels_csv")),
        demographics=load_demographics(data_path(config, "demographics_csv")),
        patient_id_width=config["dataset"]["patient_id_width"],
        max_hamming=config["qc"]["leakage_max_hamming"],
        min_correlation=config["qc"]["leakage_min_correlation"],
        skip_leakage=skip_leakage,
    )

    print("=== Quality gates ===")
    for check in report.checks:
        print(f"  [{check['result']:4s}] {check['check']}: {check['detail']}")
        for item in check["errors"][:10]:
            print(f"        ERROR   {item}")
        if len(check["errors"]) > 10:
            print(f"        ... and {len(check['errors']) - 10} more errors")
        for item in check["warnings"][:10]:
            print(f"        WARNING {item}")
        if len(check["warnings"]) > 10:
            print(f"        ... and {len(check['warnings']) - 10} more warnings")

    if leakage.get("top_candidates"):
        # Reported for human review even when below the failure threshold:
        # a gate is a decision, not the whole picture.
        print("\nClosest cross-patient pairs (review candidates, not failures):")
        for pair in leakage["top_candidates"][:5]:
            print(f"  corr={pair['correlation']:.3f} hamming={pair['hamming']:2d}  "
                  f"{pair['a']} vs {pair['b']}")
        print(f"  (anatomical similarity; threshold to fail is "
              f"{config['qc']['leakage_min_correlation']})")

    print(f"\nErrors: {report.n_errors}   Warnings: {report.n_warnings}")
    report_path = data_path(config, "artifacts_dir") / "qc_report.json"
    write_qc_report(report, report_path)
    print(f"JSON report: {report_path}")
    print(f"\nGATE: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


def cmd_release(config: dict, force: bool, verify: bool) -> int:
    """Publish (or verify) an immutable, patient-split dataset release."""
    import datetime as _dt

    version = config["release"]["version"]
    release_dir = data_path(config, "datasets_dir") / version
    curated_dir = data_path(config, "curated_dir")

    if verify:
        db_path = data_path(config, "manifest_db")
        salt = load_or_create_salt(data_path(config, "salt_file"))
        findings = verify_release(
            release_dir, curated_dir, build_pseudonym_map(db_path, salt)
        )
        print(f"=== Verifying {release_dir} ===")
        total = sum(len(v) for v in findings.values())
        print(f"Release files modified/missing: {len(findings['release_files'])}")
        print(f"Referenced images drifted:      {len(findings['referenced_images'])}")
        print(f"Real patient IDs leaked:        {len(findings['privacy'])}")
        for item in (
            findings["release_files"] + findings["referenced_images"] + findings["privacy"]
        )[:10]:
            print(f"  - {item}")
        print(f"\nVERIFY: {'PASS' if total == 0 else 'FAIL'}")
        return 0 if total == 0 else 1

    db_path = data_path(config, "manifest_db")
    if not db_path.is_file():
        print("No manifest found — run `python -m medimageforge ingest` first.")
        return 1
    if release_dir.exists() and not force:
        print(f"{release_dir} already exists. A published release is immutable.")
        print("Bump release.version in the config, or pass --force to replace it.")
        return 1

    labels_csv = data_path(config, "labels_csv")
    labels = load_labels(labels_csv)
    width = config["dataset"]["patient_id_width"]

    # Pseudonyms come from Step 6: a release leaves the controlled zone.
    salt = load_or_create_salt(data_path(config, "salt_file"))
    pseudonyms = build_pseudonym_map(db_path, salt)

    strata = patient_strata(labels, width)
    assignment = split_patients(
        strata,
        config["release"]["ratios"],
        config["release"]["seed"],
        config["release"]["stratify_by"],
    )
    assert_no_patient_overlap(assignment)

    index = build_index(db_path, assignment, pseudonyms, labels, width)
    if index.empty:
        print("No curated files found — run `python -m medimageforge curate` first.")
        return 1
    stats = split_statistics(index, strata, assignment)

    metadata = {
        "version": version,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "code_version": __version__,
        "seed": config["release"]["seed"],
        "ratios": config["release"]["ratios"],
        "stratify_by": config["release"]["stratify_by"],
        "labels_file": labels_csv.name,
        "labels_sha256": hashlib.sha256(labels_csv.read_bytes()).hexdigest(),
        "total_patients": len(assignment),
        "total_images": int(len(index)),
    }
    # Carried forward from the Step 8 QC report so a reader of the release
    # learns about them without having to dig through our artifacts.
    known_issues = [
        "Patient 084 has 36 brain slices but only 35 bone slices (slice 36 has "
        "no bone counterpart). Brain-window models are unaffected.",
        "Three patients are under 1 year old (the youngest ~1 day). Genuine "
        "data, but paediatric anatomy differs substantially from adult.",
        "Class imbalance: roughly 13% of brain slices show a hemorrhage.",
        "Masks were recovered from JPEG and thresholded to binary in curation; "
        "boundaries are approximate at the pixel level.",
        "Labels are a two-radiologist consensus with no inter-rater "
        "disagreement recorded, so annotation uncertainty cannot be estimated.",
    ]

    summary = write_release(
        release_dir, index, assignment, pseudonyms, stats, metadata, known_issues
    )

    print(f"=== Released {version} ===")
    print(f"{'split':12s} {'patients':>8s} {'w/ HGE':>7s} {'images':>7s} "
          f"{'brain':>6s} {'HGE slices':>10s} {'rate':>6s}")
    for split in ("train", "validation", "test"):
        s = stats[split]
        print(f"{split:12s} {s['patients']:8d} {s['patients_with_hemorrhage']:7d} "
              f"{s['images']:7d} {s['brain_slices']:6d} {s['hemorrhage_slices']:10d} "
              f"{s['hemorrhage_rate']:6.3f}")
    print(f"\nNo patient appears in two splits (verified).")
    print(f"Files: {', '.join(summary.files)}")
    print(f"Location: {release_dir}  (written read-only)")
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
    p_privacy = sub.add_parser(
        "privacy", help="Run the PHI gate and pseudonymize working artifacts"
    )
    p_privacy.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Scan only the first N images for metadata (0 = all)",
    )
    sub.add_parser(
        "load-labels", help="Load the labels CSV into the normalized label store"
    )
    p_show = sub.add_parser(
        "show-slice", help="Show labels + mask + provenance for one slice"
    )
    p_show.add_argument("patient", help="Patient folder name, e.g. 049")
    p_show.add_argument("slice_no", type=int, help="Slice number, e.g. 14")
    p_qc = sub.add_parser("qc", help="Run automated quality gates")
    p_qc.add_argument(
        "--skip-leakage",
        action="store_true",
        help="Skip perceptual-hash leakage detection (the slow check)",
    )
    p_release = sub.add_parser(
        "release", help="Publish an immutable, patient-split dataset version"
    )
    p_release.add_argument(
        "--force", action="store_true", help="Replace an existing release (normally refused)"
    )
    p_release.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing release against its checksums and the curated zone",
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
    if args.command == "privacy":
        return cmd_privacy(config, args.sample)
    if args.command == "load-labels":
        return cmd_load_labels(config)
    if args.command == "show-slice":
        return cmd_show_slice(config, args.patient, args.slice_no)
    if args.command == "qc":
        return cmd_qc(config, args.skip_leakage)
    if args.command == "release":
        return cmd_release(config, args.force, args.verify)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
