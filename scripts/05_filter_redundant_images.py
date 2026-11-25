
#!/usr/bin/env python3
"""
Stage 05: safely create a filtered copy of cleaned_dataset.

- Uses filter_review_images.csv
- Removes only images explicitly marked REVIEWED + REMOVE
- Copies the remaining images and labels into filtered_dataset
- Never modifies original_dataset or cleaned_dataset
"""

from pathlib import Path
import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEANED_DIR = PROJECT_DIR / "cleaned_dataset"
REVIEW_CSV = (
    PROJECT_DIR / "reports" / "duplicate_review"
    / "filter_review_images.csv"
)
MANIFEST_CSV = (
    PROJECT_DIR / "reports" / "split_cleanup"
    / "image_manifest.csv"
)
OUTPUT_DIR = PROJECT_DIR / "filtered_dataset"
REPORT_DIR = PROJECT_DIR / "reports" / "filtering"


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def label_class_counts(dataset_dir):
    counts = Counter()

    for split in ("train", "valid", "test"):
        label_dir = dataset_dir / split / "labels"

        if not label_dir.exists():
            continue

        for label in label_dir.glob("*.txt"):
            for line in label.read_text(
                encoding="utf-8-sig", errors="replace"
            ).splitlines():

                parts = line.split()

                if not parts:
                    continue

                try:
                    counts[int(float(parts[0]))] += 1
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Unrecognized label row in {label}: {line}"
                    )

    return counts


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create filtered_dataset"
    )

    args = parser.parse_args()

    if not CLEANED_DIR.exists():
        raise FileNotFoundError(
            f"cleaned_dataset not found: {CLEANED_DIR}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Rename it manually; this script will not overwrite it."
        )

    review_rows = read_csv(REVIEW_CSV)
    manifest_rows = read_csv(MANIFEST_CSV)

    split_by_image = {
        r["image"]: r["new_split"]
        for r in manifest_rows
    }

    remove_images = set()
    group_splits = defaultdict(set)
    seen = set()

    for row in review_rows:

        image = row["image"]

        if image in seen:
            raise ValueError(
                f"Duplicate image row in review CSV: {image}"
            )

        seen.add(image)

        decision = (
            row.get("decision") or "KEEP"
        ).strip().upper()

        status = (
            row.get("review_status") or "PENDING"
        ).strip().upper()

        reason = (row.get("reason") or "").strip()

        if decision not in {"KEEP", "REMOVE"}:
            raise ValueError(
                f"Invalid decision for {image}: {decision}"
            )

        if status not in {"PENDING", "REVIEWED"}:
            raise ValueError(
                f"Invalid review_status for {image}: {status}"
            )

        if decision == "REMOVE":

            if status != "REVIEWED":
                raise ValueError(
                    f"{image}: REMOVE requires REVIEWED status"
                )

            if not reason:
                raise ValueError(
                    f"{image}: add a reason before removing it"
                )

            remove_images.add(image)

        split = split_by_image.get(image)

        if not split:
            raise ValueError(
                f"{image} is missing from image_manifest.csv"
            )

        group_splits[row["group_id"]].add(split)

    # Make sure every duplicate group belongs to one split.
    conflicts = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) != 1
    }

    if conflicts:
        raise ValueError(
            f"Groups appear in multiple cleaned splits: {conflicts}"
        )

    # Validate reviewed images and their paired labels.
    for image in seen:

        split = split_by_image[image]

        img_path = (
            CLEANED_DIR / split / "images" / image
        )

        label_path = (
            CLEANED_DIR / split / "labels"
            / (Path(image).stem + ".txt")
        )

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image missing: {img_path}"
            )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Paired label missing: {label_path}"
            )

    keep_counts = Counter()
    planned_images = []

    # Prepare the list of images that will be copied.
    for split in ("train", "valid", "test"):

        image_dir = CLEANED_DIR / split / "images"
        label_dir = CLEANED_DIR / split / "labels"

        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(
                f"Missing images/labels directory for {split}"
            )

        for img_path in sorted(image_dir.iterdir()):

            if not img_path.is_file():
                continue

            label_path = (
                label_dir / (img_path.stem + ".txt")
            )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing paired label for {img_path}"
                )

            if img_path.name in remove_images:
                continue

            planned_images.append(
                (split, img_path, label_path)
            )

            keep_counts[split] += 1

    before_counts = {
        split: len(
            list((CLEANED_DIR / split / "images").glob("*"))
        )
        for split in ("train", "valid", "test")
    }

    print("\nFILTER PREVIEW")
    print("--------------")

    print(
        f"Images marked REMOVE and reviewed: "
        f"{len(remove_images)}"
    )

    for split in ("train", "valid", "test"):

        removed_here = (
            before_counts[split] - keep_counts[split]
        )

        print(
            f"{split}: {before_counts[split]} "
            f"-> {keep_counts[split]} "
            f"(remove {removed_here})"
        )

    # Compare class counts before and after filtering.
    before_classes = label_class_counts(CLEANED_DIR)
    after_classes = Counter()

    for _, _, label_path in planned_images:

        for line in label_path.read_text(
            encoding="utf-8-sig", errors="replace"
        ).splitlines():

            parts = line.split()

            if parts:
                after_classes[int(float(parts[0]))] += 1

    print("\nClass-instance counts (before -> after):")

    for class_id in sorted(
        set(before_classes) | set(after_classes)
    ):

        print(
            f"Class {class_id}: "
            f"{before_classes[class_id]} "
            f"-> {after_classes[class_id]}"
        )

    missing_classes = [
        c for c, count in before_classes.items()
        if count > 0 and after_classes[c] == 0
    ]

    if missing_classes:
        raise ValueError(
            "Filtering would remove every annotation "
            f"for class IDs: {missing_classes}"
        )

    # Default mode: preview only.
    if not args.apply:

        print("\nDry run only.")
        print("No files were created or changed.")
        print("If the preview is correct, run again with --apply.")

        return

    temp_dir = PROJECT_DIR / "filtered_dataset_build"

    if temp_dir.exists():
        raise FileExistsError(
            f"Temporary build folder already exists: {temp_dir}"
        )

    temp_dir.mkdir(parents=True)

    try:

        for split in ("train", "valid", "test"):

            (temp_dir / split / "images").mkdir(
                parents=True
            )

            (temp_dir / split / "labels").mkdir(
                parents=True
            )

        # Copy only the selected images and their labels.
        for split, img_path, label_path in planned_images:

            shutil.copy2(
                img_path,
                temp_dir / split / "images" / img_path.name
            )

            shutil.copy2(
                label_path,
                temp_dir / split / "labels" / label_path.name
            )

        # Copy dataset configuration.
        for filename in ("data.yaml", "README.md"):

            source = CLEANED_DIR / filename

            if source.exists():
                shutil.copy2(
                    source,
                    temp_dir / filename
                )

        # Verify the four class names are preserved.
        yaml_path = temp_dir / "data.yaml"

        if yaml_path.exists():

            yaml_text = yaml_path.read_text(
                encoding="utf-8-sig"
            )

            expected = (
                "['Bottle', 'Detached_Cap', "
                "'With_Cap', 'Without_Cap']"
            )

            if expected not in yaml_text.replace('"', "'"):
                raise ValueError(
                    "data.yaml class names differ from "
                    "the expected four classes."
                )

        temp_dir.rename(OUTPUT_DIR)

    except Exception:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise

    # Save filtering report.
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "source": "cleaned_dataset",
        "output": "filtered_dataset",
        "removed_images": sorted(remove_images),
        "removed_count": len(remove_images),
        "kept_counts_by_split": dict(keep_counts),
        "class_instance_counts_before": {
            str(k): v for k, v in before_classes.items()
        },
        "class_instance_counts_after": {
            str(k): v for k, v in after_classes.items()
        },
        "class_names_preserved": [
            "Bottle",
            "Detached_Cap",
            "With_Cap",
            "Without_Cap"
        ],
        "original_dataset_modified": False,
        "cleaned_dataset_modified": False
    }

    report_path = REPORT_DIR / "filtering_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nFiltering completed successfully!")
    print(f"New dataset: {OUTPUT_DIR}")
    print("original_dataset and cleaned_dataset were not modified.")


if __name__ == "__main__":
    main()
