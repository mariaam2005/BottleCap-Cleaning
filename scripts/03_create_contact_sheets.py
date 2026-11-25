
"""Create contact sheets for cross-split near-duplicate images."""

from pathlib import Path
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def find_image(dataset_dir, split, filename):
    filename = Path(str(filename)).name

    # Search in the expected split first
    path = dataset_dir / str(split).lower() / "images" / filename
    if path.is_file():
        return path

    # Fallback: search all splits
    for folder in ["train", "valid", "test"]:
        path = dataset_dir / folder / "images" / filename
        if path.is_file():
            return path

    return None


def load_image(path, size=(320, 230)):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail(size)

        canvas = Image.new("RGB", size, "white")
        canvas.paste(
            image,
            ((size[0] - image.width) // 2,
             (size[1] - image.height) // 2)
        )
        return canvas


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1]
    )

    parser.add_argument("--max-distance", type=int, default=2)
    parser.add_argument("--pairs-per-sheet", type=int, default=6)

    args = parser.parse_args()

    project = args.project.resolve()

    dataset_dir = project / "original_dataset"
    report_path = project / "reports" / "near_duplicates.csv"

    output_dir = project / "reports" / "duplicate_review"
    sheets_dir = output_dir / "contact_sheets"

    sheets_dir.mkdir(parents=True, exist_ok=True)

    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    df = pd.read_csv(report_path)

    required = {
        "image_1", "split_1",
        "image_2", "split_2",
        "phash_distance"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["phash_distance"] = pd.to_numeric(
        df["phash_distance"], errors="coerce"
    )

    if "cross_split" in df.columns:
        cross = df["cross_split"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    else:
        cross = (
            df["split_1"].astype(str).str.lower()
            != df["split_2"].astype(str).str.lower()
        )

    review = df[
        cross & (df["phash_distance"] <= args.max_distance)
    ].copy()

    review = review.drop_duplicates(
        subset=["image_1", "image_2"]
    ).reset_index(drop=True)

    if review.empty:
        print("No pairs found for review.")
        return

    records = []

    for index, row in review.iterrows():
        path1 = find_image(
            dataset_dir, row["split_1"], row["image_1"]
        )
        path2 = find_image(
            dataset_dir, row["split_2"], row["image_2"]
        )

        records.append({
            "pair_id": index + 1,
            "image_1": Path(str(row["image_1"])).name,
            "split_1": row["split_1"],
            "image_2": Path(str(row["image_2"])).name,
            "split_2": row["split_2"],
            "phash_distance": int(row["phash_distance"]),
            "image_1_found": path1 is not None,
            "image_2_found": path2 is not None
        })

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(
        output_dir / "pairs_to_review.csv",
        index=False
    )

    per_sheet = max(1, args.pairs_per_sheet)

    pair_width = 340
    image_height = 230
    header_height = 28
    footer_height = 28
    sheet_height = header_height + image_height + footer_height

    for start in range(0, len(review), per_sheet):

        batch = review.iloc[start:start + per_sheet]

        sheet = Image.new(
            "RGB",
            (pair_width * 2, sheet_height * len(batch)),
            "#eeeeee"
        )

        draw = ImageDraw.Draw(sheet)

        for offset, (_, row) in enumerate(batch.iterrows()):

            pair_number = start + offset + 1
            y = offset * sheet_height

            distance = int(row["phash_distance"])

            draw.text(
                (8, y + 7),
                f"Pair {pair_number} | Distance: {distance} | "
                f"{row['split_1']} vs {row['split_2']}",
                fill="black"
            )

            for side, (image_col, split_col) in enumerate([
                ("image_1", "split_1"),
                ("image_2", "split_2")
            ]):

                x = side * pair_width

                image_path = find_image(
                    dataset_dir,
                    row[split_col],
                    row[image_col]
                )

                if image_path:
                    try:
                        tile = load_image(image_path)
                        sheet.paste(
                            tile,
                            (x + 10, y + header_height)
                        )

                        label = (
                            f"{row[split_col]} | "
                            f"{Path(str(row[image_col])).name}"
                        )

                    except Exception as error:
                        label = f"Could not open image: {error}"

                else:
                    label = (
                        f"IMAGE NOT FOUND: "
                        f"{Path(str(row[image_col])).name}"
                    )

                    draw.rectangle(
                        (
                            x + 10,
                            y + header_height,
                            x + 330,
                            y + header_height + image_height
                        ),
                        fill="#ffdddd",
                        outline="red"
                    )

                    draw.text(
                        (x + 18, y + header_height + 100),
                        "IMAGE NOT FOUND",
                        fill="red"
                    )

                draw.rectangle(
                    (
                        x + 9,
                        y + header_height - 1,
                        x + 331,
                        y + header_height + image_height
                    ),
                    outline="#888888"
                )

                draw.text(
                    (x + 10, y + header_height + image_height + 5),
                    label[:48],
                    fill="black"
                )

        filename = (
            f"review_{start + 1:04d}_to_"
            f"{start + len(batch):04d}.jpg"
        )

        sheet.save(sheets_dir / filename, quality=90)

    missing_count = sum(
        not r["image_1_found"] or not r["image_2_found"]
        for r in records
    )

    print(f"Done: {len(review)} pairs selected.")
    print(f"Contact sheets: {sheets_dir}")
    print(f"Review CSV: {output_dir / 'pairs_to_review.csv'}")
    print(f"Pairs with missing images: {missing_count}")
    print("Original dataset was NOT modified.")


if __name__ == "__main__":
    main()
