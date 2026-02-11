"""
Tile a YOLO-format dataset into overlapping patches and remap labels.

- Expects images/ and labels/ with train/val/test subfolders under the source root.
- Produces tiled images/labels under the destination root with the same split names.
- Boxes are clipped to tile bounds; boxes with visible area fraction below --min_visible are dropped.
- Enforces a minimum box side length of 4 px after clipping.
- Keeps class IDs as-is (single-class assumed but not required).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image


def generate_tile_origins(size: int, tile: int, overlap: int) -> List[int]:
    """Generate tile start positions covering [0, size) with overlap, ending flush to the border."""
    stride = max(tile - overlap, 1)
    origins = list(range(0, size, stride))
    if origins and origins[-1] + tile < size:
        origins.append(max(size - tile, 0))
    return sorted(set(origins))


def parse_label_file(path: Path, img_w: int, img_h: int) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def clip_and_normalize_box(
    box: Tuple[int, float, float, float, float],
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    min_visible: float,
    min_side_px: float = 4.0,
) -> Tuple[int, float, float, float, float] | None:
    cls, x1, y1, x2, y2 = box
    inter_x1 = max(x1, tile_x)
    inter_y1 = max(y1, tile_y)
    inter_x2 = min(x2, tile_x + tile_w)
    inter_y2 = min(y2, tile_y + tile_h)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return None

    orig_area = (x2 - x1) * (y2 - y1)
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    if orig_area <= 0:
        return None
    if inter_area / orig_area < min_visible:
        return None

    # Enforce minimum side length inside the tile.
    inter_w = max(inter_x2 - inter_x1, min_side_px)
    inter_h = max(inter_y2 - inter_y1, min_side_px)

    # Clip again if min-side expansion pushed it out of bounds.
    inter_x2 = min(inter_x1 + inter_w, tile_x + tile_w)
    inter_y2 = min(inter_y1 + inter_h, tile_y + tile_h)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    if inter_w <= 0 or inter_h <= 0:
        return None

    cx = ((inter_x1 + inter_x2) / 2.0 - tile_x) / tile_w
    cy = ((inter_y1 + inter_y2) / 2.0 - tile_y) / tile_h
    w = inter_w / tile_w
    h = inter_h / tile_h

    # Clamp to [0,1]
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    if w == 0 or h == 0:
        return None
    return cls, cx, cy, w, h


def tile_image(
    img_path: Path,
    labels_path: Path,
    dst_images: Path,
    dst_labels: Path,
    tile: int,
    overlap: int,
    min_visible: float,
    keep_empty: bool,
) -> int:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        if "exif" in img.info:
            img.info.pop("exif", None)
        w, h = img.size
        boxes = parse_label_file(labels_path, w, h)

        x_starts = generate_tile_origins(w, tile, overlap)
        y_starts = generate_tile_origins(h, tile, overlap)

        kept_tiles = 0
        for y0 in y_starts:
            for x0 in x_starts:
                tile_w = min(tile, w - x0)
                tile_h = min(tile, h - y0)
                tile_boxes: List[Tuple[int, float, float, float, float]] = []
                for box in boxes:
                    clipped = clip_and_normalize_box(box, x0, y0, tile_w, tile_h, min_visible)
                    if clipped:
                        tile_boxes.append(clipped)
                if not tile_boxes and not keep_empty:
                    continue

                tile_img = img.crop((x0, y0, x0 + tile_w, y0 + tile_h))
                tile_name = f"{img_path.stem}_x{x0}_y{y0}.jpg"
                tile_img.save(dst_images / tile_name, format="JPEG", quality=95)
                label_lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in tile_boxes]
                (dst_labels / f"{Path(tile_name).stem}.txt").write_text("\n".join(label_lines))
                kept_tiles += 1
        return kept_tiles


def process_split(
    split: str,
    src_root: Path,
    dst_root: Path,
    tile: int,
    overlap: int,
    min_visible: float,
    keep_empty_non_train: bool,
) -> int:
    src_images = src_root / "images" / split
    src_labels = src_root / "labels" / split
    dst_images = dst_root / "images" / split
    dst_labels = dst_root / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    for img_path in sorted(src_images.glob("*.jpg")):
        labels_path = src_labels / f"{img_path.stem}.txt"
        keep_empty = keep_empty_non_train and split != "train"
        total_tiles += tile_image(
            img_path,
            labels_path,
            dst_images,
            dst_labels,
            tile,
            overlap,
            min_visible,
            keep_empty,
        )
    return total_tiles


def save_dataset_yaml(dst_root: Path) -> None:
    yaml_path = dst_root / "dataset.yaml"
    yaml_content = (
        "# Auto-generated tiled YOLO dataset file\n"
        f"path: {dst_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: parasite\n"
    )
    yaml_path.write_text(yaml_content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile a YOLO dataset and remap labels")
    parser.add_argument("--src", type=Path, default=Path("data/Processed/yolo_malaria"))
    parser.add_argument("--dst", type=Path, default=Path("data/Processed/yolo_malaria_tiles"))
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--min_visible", type=float, default=0.2, help="Min visible area fraction to keep a box")
    parser.add_argument(
        "--keep_empty_non_train",
        action="store_true",
        help="Keep empty tiles for val/test; train still drops empty tiles",
    )
    args = parser.parse_args()

    dst_root: Path = args.dst
    dst_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ("train", "val", "test"):
        tiles = process_split(
            split,
            args.src,
            dst_root,
            args.tile_size,
            args.overlap,
            args.min_visible,
            args.keep_empty_non_train,
        )
        print(f"{split}: {tiles} tiles")
        total += tiles
    save_dataset_yaml(dst_root)
    print(f"Total tiles written: {total}")
    print(f"Dataset YAML: {(dst_root / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()
