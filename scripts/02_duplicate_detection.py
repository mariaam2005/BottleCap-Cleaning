
from pathlib import Path
from collections import defaultdict
import hashlib
import csv
import cv2
import imagehash
from PIL import Image
from tqdm import tqdm

# ==========================================
# PATHS & SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "original_dataset"
REPORTS = ROOT / "reports"

REPORTS.mkdir(exist_ok=True)

SPLITS = ["train", "valid", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Lower distance = more similar
PHASH_THRESHOLD = 6

# ==========================================
# COLLECT IMAGES
# ==========================================

records = []

for split in SPLITS:

    image_dir = DATASET / split / "images"

    for path in image_dir.iterdir():

        if path.is_file() and path.suffix.lower() in EXTENSIONS:

            records.append({
                "path": path,
                "name": path.name,
                "split": split,
                "sha256": None,
                "phash": None
            })

print(f"\nTotal images: {len(records)}")

# ==========================================
# HASH CALCULATION
# ==========================================

print("\nCalculating image hashes...")

for record in tqdm(records):

    path = record["path"]

    # Exact hash
    sha = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)

    record["sha256"] = sha.hexdigest()

    # Perceptual hash
    with Image.open(path) as image:
        image = image.convert("RGB")
        record["phash"] = imagehash.phash(image)

# ==========================================
# EXACT DUPLICATES
# ==========================================

exact_groups = defaultdict(list)

for record in records:
    exact_groups[record["sha256"]].append(record)

exact_duplicates = []

for group in exact_groups.values():

    if len(group) > 1:

        for record in group:
            exact_duplicates.append(record)

# ==========================================
# NEAR DUPLICATES
# ==========================================

print("\nChecking similar images...")

near_duplicates = []

for i in tqdm(range(len(records))):

    for j in range(i + 1, len(records)):

        first = records[i]
        second = records[j]

        # Skip exact duplicates
        if first["sha256"] == second["sha256"]:
            continue

        distance = first["phash"] - second["phash"]

        if distance <= PHASH_THRESHOLD:

            near_duplicates.append({
                "image_1": first["name"],
                "split_1": first["split"],
                "image_2": second["name"],
                "split_2": second["split"],
                "phash_distance": distance,
                "cross_split": first["split"] != second["split"]
            })

# ==========================================
# SAVE EXACT DUPLICATE REPORT
# ==========================================

exact_path = REPORTS / "exact_duplicates.csv"

with open(exact_path, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "split", "sha256"]
    )

    writer.writeheader()

    for record in exact_duplicates:

        writer.writerow({
            "filename": record["name"],
            "split": record["split"],
            "sha256": record["sha256"]
        })

# ==========================================
# SAVE NEAR DUPLICATE REPORT
# ==========================================

near_path = REPORTS / "near_duplicates.csv"

fields = [
    "image_1",
    "split_1",
    "image_2",
    "split_2",
    "phash_distance",
    "cross_split"
]

with open(near_path, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.DictWriter(file, fieldnames=fields)

    writer.writeheader()
    writer.writerows(near_duplicates)

# ==========================================
# FINAL SUMMARY
# ==========================================

cross_split_pairs = [
    pair for pair in near_duplicates
    if pair["cross_split"]
]

print("\n========== DUPLICATE REPORT ==========")

print("Total images:", len(records))
print("Exact duplicate image entries:", len(exact_duplicates))
print("Exact duplicate groups:",
      sum(1 for g in exact_groups.values() if len(g) > 1))

print("Near duplicate pairs:", len(near_duplicates))
print("Cross-split similar pairs:", len(cross_split_pairs))

print("\nReports saved:")
print(exact_path)
print(near_path)

print("\nNo images were deleted or modified.")
print("Duplicate detection completed!")
