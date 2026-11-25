
from pathlib import Path
from collections import Counter
import shutil
import csv
import re

# ==================================================
# PATHS & SETTINGS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_DIR / "original_dataset"
OUTPUT_DIR = PROJECT_DIR / "cleaned_dataset"

GROUPS_FILE = (
    PROJECT_DIR
    / "reports"
    / "duplicate_review"
    / "group_membership.csv"
)

REPORT_DIR = PROJECT_DIR / "reports" / "split_cleanup"

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ==================================================
# READ GROUP MEMBERSHIP
# ==================================================

def read_groups():
    if not GROUPS_FILE.exists():
        raise FileNotFoundError(
            f"Group membership file not found:\n{GROUPS_FILE}"
        )

    groups = {}

    with open(GROUPS_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"group_id", "split", "image"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain these columns: {required}"
            )

        seen_images = set()

        for row in reader:
            group_id = row["group_id"].strip()
            split = row["split"].strip().lower()
            image = Path(row["image"].strip()).name

            if split not in SPLITS:
                raise ValueError(
                    f"Invalid split '{split}' for {image}"
                )

            if image in seen_images:
                raise ValueError(
                    f"Image appears more than once in CSV: {image}"
                )

            seen_images.add(image)

            groups.setdefault(group_id, []).append({
                "image": image,
                "original_split": split
            })

    return groups


# ==================================================
# ASSIGN ONE SPLIT TO EACH GROUP
# ==================================================

def assign_group_splits(groups):
    assignments = {}

    # Majority split wins.
    # Ties are resolved using train > valid > test.
    tie_priority = {
        "train": 0,
        "valid": 1,
        "test": 2
    }

    for group_id, members in groups.items():

        counts = Counter(
            member["original_split"] for member in members
        )

        max_count = max(counts.values())

        candidates = [
            split for split in SPLITS
            if counts[split] == max_count
        ]

        destination = min(
            candidates,
            key=lambda split: tie_priority[split]
        )

        assignments[group_id] = destination

    return assignments


# ==================================================
# VALIDATE ORIGINAL DATASET
# ==================================================

def validate_dataset(groups):

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Original dataset not found:\n{SOURCE_DIR}"
        )

    image_locations = {}
    label_locations = {}

    for split in SPLITS:

        image_dir = SOURCE_DIR / split / "images"
        label_dir = SOURCE_DIR / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing images or labels folder in {split}"
            )

        for image_path in image_dir.iterdir():

            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:

                if image_path.name in image_locations:
                    raise ValueError(
                        f"Duplicate filename in original dataset: {image_path.name}"
                    )

                image_locations[image_path.name] = split

                label_path = label_dir / f"{image_path.stem}.txt"

                if not label_path.is_file():
                    raise FileNotFoundError(
                        f"Missing label for image:\n{image_path}"
                    )

                label_locations[image_path.name] = label_path

    # Check that every CSV image exists in its reported original split.
    for group_id, members in groups.items():

        for member in members:

            image = member["image"]
            expected_split = member["original_split"]

            if image not in image_locations:
                raise FileNotFoundError(
                    f"Image from {group_id} not found: {image}"
                )

            actual_split = image_locations[image]

            if actual_split != expected_split:
                raise ValueError(
                    f"Split mismatch for {image}: "
                    f"CSV says {expected_split}, "
                    f"dataset says {actual_split}"
                )

    return image_locations, label_locations


# ==================================================
# PREPARE IMAGE DESTINATIONS
# ==================================================

def build_image_plan(groups, assignments, image_locations):

    image_plan = {}

    group_lookup = {}

    for group_id, members in groups.items():

        destination = assignments[group_id]

        for member in members:

            image = member["image"]

            image_plan[image] = destination
            group_lookup[image] = group_id

    # Images not included in any group remain in their original split.
    for image, original_split in image_locations.items():

        if image not in image_plan:
            image_plan[image] = original_split
            group_lookup[image] = "UNGROUPED"

    return image_plan, group_lookup


# ==================================================
# COPY DATASET WITHOUT MODIFYING ORIGINAL
# ==================================================

def create_cleaned_dataset(
    image_locations,
    label_locations,
    image_plan
):

    temp_dir = PROJECT_DIR / "cleaned_dataset_build"

    if temp_dir.exists():
        raise FileExistsError(
            f"Temporary output already exists:\n{temp_dir}\n"
            "Inspect it before removing or renaming it."
        )

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty:\n{OUTPUT_DIR}\n"
            "Nothing was overwritten."
        )

    temp_dir.mkdir()

    try:

        for split in SPLITS:
            (temp_dir / split / "images").mkdir(parents=True)
            (temp_dir / split / "labels").mkdir(parents=True)

        for image, original_split in image_locations.items():

            destination_split = image_plan[image]

            source_image = (
                SOURCE_DIR / original_split / "images" / image
            )

            source_label = label_locations[image]

            destination_image = (
                temp_dir / destination_split / "images" / image
            )

            destination_label = (
                temp_dir / destination_split / "labels"
                / f"{Path(image).stem}.txt"
            )

            shutil.copy2(source_image, destination_image)
            shutil.copy2(source_label, destination_label)

        # Copy and update data.yaml while preserving class names/settings.
        source_yaml = SOURCE_DIR / "data.yaml"

        if not source_yaml.exists():
            raise FileNotFoundError(
                f"data.yaml not found:\n{source_yaml}"
            )

        yaml_text = source_yaml.read_text(encoding="utf-8")

        yaml_text = re.sub(
            r"(?m)^\s*path\s*:.*\n?", "", yaml_text
        )

        yaml_text = re.sub(
            r"(?m)^\s*train\s*:.*\n?", "", yaml_text
        )

        yaml_text = re.sub(
            r"(?m)^\s*val\s*:.*\n?", "", yaml_text
        )

        yaml_text = re.sub(
            r"(?m)^\s*test\s*:.*\n?", "", yaml_text
        )

        yaml_text = (
            "path: .\n"
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n\n"
            + yaml_text.lstrip()
        )

        (temp_dir / "data.yaml").write_text(
            yaml_text, encoding="utf-8"
        )

        # Preserve the original Roboflow README if available.
        readme = SOURCE_DIR / "README.roboflow.txt"

        if readme.exists():
            shutil.copy2(readme, temp_dir / readme.name)

        # Verify copied image and label counts.
        for split in SPLITS:

            images = list(
                (temp_dir / split / "images").iterdir()
            )

            labels = list(
                (temp_dir / split / "labels").glob("*.txt")
            )

            if len(images) != len(labels):
                raise RuntimeError(
                    f"Image/label count mismatch in {split}: "
                    f"{len(images)} images, {len(labels)} labels"
                )

        # Replace only an existing EMPTY output directory.
        if OUTPUT_DIR.exists():
            OUTPUT_DIR.rmdir()

        temp_dir.rename(OUTPUT_DIR)

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


# ==================================================
# GENERATE REPORTS
# ==================================================

def generate_reports(
    groups,
    assignments,
    image_locations,
    image_plan,
    group_lookup
):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Group assignment report.
    with open(
        REPORT_DIR / "group_assignments.csv",
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "group_id",
            "original_train",
            "original_valid",
            "original_test",
            "total_images",
            "assigned_split",
            "images_to_reassign"
        ])

        for group_id, members in groups.items():

            counts = Counter(
                member["original_split"] for member in members
            )

            destination = assignments[group_id]

            moved = sum(
                member["original_split"] != destination
                for member in members
            )

            writer.writerow([
                group_id,
                counts["train"],
                counts["valid"],
                counts["test"],
                len(members),
                destination,
                moved
            ])

    # Per-image movement manifest.
    with open(
        REPORT_DIR / "image_manifest.csv",
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "image",
            "group_id",
            "original_split",
            "new_split"
        ])

        for image, original_split in image_locations.items():

            writer.writerow([
                image,
                group_lookup[image],
                original_split,
                image_plan[image]
            ])

    # Before/after summary.
    before = Counter(image_locations.values())
    after = Counter(image_plan.values())

    with open(
        REPORT_DIR / "split_summary.csv",
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "split",
            "before_cleanup",
            "after_cleanup"
        ])

        for split in SPLITS:
            writer.writerow([
                split,
                before[split],
                after[split]
            ])

    print("\n========== SPLIT CLEANUP SUMMARY ==========")

    for split in SPLITS:
        print(
            f"{split.upper():5} | "
            f"Before: {before[split]:4} | "
            f"After: {after[split]:4}"
        )

    print("\nGroups processed:", len(groups))
    print("Images processed:", len(image_locations))

    print("\nReports saved in:")
    print(REPORT_DIR)

    print("\nCleaned dataset saved in:")
    print(OUTPUT_DIR)

    print("\nOriginal dataset was NOT modified.")


# ==================================================
# MAIN
# ==================================================

def main():

    print("Starting Split Cleanup...\n")

    groups = read_groups()

    assignments = assign_group_splits(groups)

    image_locations, label_locations = validate_dataset(groups)

    image_plan, group_lookup = build_image_plan(
        groups,
        assignments,
        image_locations
    )

    print("Validation successful.")
    print("Groups:", len(groups))
    print("Images:", len(image_locations))

    create_cleaned_dataset(
        image_locations,
        label_locations,
        image_plan
    )

    generate_reports(
        groups,
        assignments,
        image_locations,
        image_plan,
        group_lookup
    )

    print("\nSplit Cleanup completed successfully!")


if __name__ == "__main__":
    main()
