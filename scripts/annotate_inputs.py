import pathlib
from typing import Dict, Tuple
from PIL import Image, ImageDraw

ROOT = pathlib.Path("data/Processed/yolo_malaria")
SPLITS = ["train", "val", "test"]
MAX_PATIENTS = 8  # limit for quick visualization
OUTPUT_DIR = pathlib.Path("results/yolo11n_long_samples/annotated_inputs")


def collect_one_per_patient() -> Dict[str, Tuple[str, pathlib.Path]]:
    picked: Dict[str, Tuple[str, pathlib.Path]] = {}
    for split in SPLITS:
        images_dir = ROOT / "images" / split
        for img_path in sorted(images_dir.glob("*.jpg")):
            prefix = img_path.stem[:8]
            if prefix not in picked:
                picked[prefix] = (split, img_path)
            if len(picked) >= MAX_PATIENTS:
                return picked
    return picked


def draw_boxes(img_path: pathlib.Path, labels_path: pathlib.Path, save_path: pathlib.Path) -> int:
    # Important: drop EXIF orientation so we draw on the raw pixel grid (see dataset note)
    img = Image.open(img_path)
    img = img.convert("RGB")
    if "exif" in img.info:
        del img.info["exif"]
    draw = ImageDraw.Draw(img)
    w, h = img.size

    box_count = 0
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            box_count += 1

    footer = f"{img_path.name} | boxes={box_count}"
    draw.text((5, 5), footer, fill="red")
    img.save(save_path, quality=90)
    return box_count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    picked = collect_one_per_patient()
    print(f"Selected {len(picked)} patients (limited to {MAX_PATIENTS}).")

    for prefix, (split, img_path) in picked.items():
        labels_path = ROOT / "labels" / split / f"{img_path.stem}.txt"
        save_path = OUTPUT_DIR / f"{img_path.stem}_annotated.jpg"
        boxes = draw_boxes(img_path, labels_path, save_path)
        print(f"{prefix} [{split}] -> {save_path} (boxes: {boxes})")


if __name__ == "__main__":
    main()
