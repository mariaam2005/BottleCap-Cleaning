
from pathlib import Path
import cv2
import yaml
import csv
import json
from collections import Counter

# ==========================================
# 1. PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "original_dataset"
REPORTS = ROOT / "reports"

REPORTS.mkdir(exist_ok=True)

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ==========================================
# 2. LOAD DATA YAML
# ==========================================

yaml_path = DATASET / "data.yaml"

with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

class_names = config.get("names", [])

if isinstance(class_names, dict):
    class_names = [
        class_names[k] for k in sorted(class_names)
    ]

print("\n========== DATASET INSPECTION ==========")
print("Dataset:", DATASET)
print("Classes:", class_names)

# ==========================================
# 3. INSPECTION
# ==========================================

summary = {}
issues = []
total_images = 0
total_annotations = 0
global_class_counts = Counter()

for split in SPLITS:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    print(f"\n========== {split.upper()} ==========")

    if not image_dir.exists() or not label_dir.exists():
        issues.append({
            "split": split,
            "file": "",
            "issue": "Missing images or labels directory"
        })
        continue

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    split_stats = {
        "images": len(images),
        "label_files": len(labels),
        "corrupted_images": 0,
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "empty_labels": 0,
        "bbox_annotations": 0,
        "polygon_annotations": 0,
        "invalid_annotations": 0,
        "class_counts": Counter()
    }

    # Check missing labels
    for name in missing_labels:
        issues.append({
            "split": split,
            "file": name,
            "issue": "Missing label file"
        })

    # Check orphan labels
    for name in orphan_labels:
        issues.append({
            "split": split,
            "file": name,
            "issue": "Label without matching image"
        })

    # Check images
    for img_path in images:

        image = cv2.imread(str(img_path))

        if image is None:
            split_stats["corrupted_images"] += 1

            issues.append({
                "split": split,
                "file": img_path.name,
                "issue": "Corrupted or unreadable image"
            })

    # Check annotations
    for label_path in labels:

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

        except Exception as error:
            issues.append({
                "split": split,
                "file": label_path.name,
                "issue": f"Cannot read label: {error}"
            })
            continue

        valid_lines = [line.strip() for line in lines if line.strip()]

        if not valid_lines:
            split_stats["empty_labels"] += 1

            issues.append({
                "split": split,
                "file": label_path.name,
                "issue": "Empty label file"
            })
            continue

        for line_number, line in enumerate(valid_lines, 1):

            try:
                values = [float(x) for x in line.split()]

                if len(values) < 5:
                    raise ValueError("Less than 5 values")

                class_id = int(values[0])

                if values[0] != class_id:
                    raise ValueError("Non-integer class ID")

                if class_id < 0 or class_id >= len(class_names):
                    raise ValueError("Class ID outside YAML range")

                coords = values[1:]

                if len(coords) == 4:
                    split_stats["bbox_annotations"] += 1

                    x, y, w, h = coords

                    if w <= 0 or h <= 0:
                        raise ValueError("Invalid bbox width/height")

                    if not all(0 <= v <= 1 for v in coords):
                        raise ValueError("BBox coordinates outside [0,1]")

                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    split_stats["polygon_annotations"] += 1

                    if not all(0 <= v <= 1 for v in coords):
                        raise ValueError("Polygon coordinates outside [0,1]")

                else:
                    raise ValueError("Invalid annotation format")

                split_stats["class_counts"][class_names[class_id]] += 1
                global_class_counts[class_names[class_id]] += 1
                total_annotations += 1

            except Exception as error:

                split_stats["invalid_annotations"] += 1

                issues.append({
                    "split": split,
                    "file": label_path.name,
                    "issue": f"Line {line_number}: {error}"
                })

    total_images += len(images)

    split_stats["class_counts"] = dict(
        split_stats["class_counts"]
    )

    summary[split] = split_stats

    print("Images:", split_stats["images"])
    print("Labels:", split_stats["label_files"])
    print("Corrupted images:", split_stats["corrupted_images"])
    print("Missing labels:", split_stats["missing_labels"])
    print("Orphan labels:", split_stats["orphan_labels"])
    print("Empty labels:", split_stats["empty_labels"])
    print("Bounding boxes:", split_stats["bbox_annotations"])
    print("Polygons:", split_stats["polygon_annotations"])
    print("Invalid annotations:", split_stats["invalid_annotations"])
    print("Class distribution:", split_stats["class_counts"])

# ==========================================
# 4. SAVE REPORTS
# ==========================================

summary_data = {
    "total_images": total_images,
    "total_annotations": total_annotations,
    "classes": class_names,
    "global_class_counts": dict(global_class_counts),
    "splits": summary,
    "total_issues": len(issues)
}

summary_path = REPORTS / "inspection_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary_data, file, indent=4)

issues_path = REPORTS / "inspection_issues.csv"

with open(issues_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["split", "file", "issue"]
    )

    writer.writeheader()
    writer.writerows(issues)

# ==========================================
# 5. FINAL RESULT
# ==========================================

print("\n========== FINAL SUMMARY ==========")
print("Total images:", total_images)
print("Total annotations:", total_annotations)
print("Total issues:", len(issues))
print("Class counts:", dict(global_class_counts))

print("\nReports saved:")
print(summary_path)
print(issues_path)

print("\nInspection completed!")
