"""Utilities for the object-detection labs.

The functions download the Penn-Fudan Pedestrian dataset, convert its instance
masks to YOLO detection labels, validate the result, and make a separate
augmented training dataset. Boxes use normalized YOLO ``x_center, y_center,
width, height`` coordinates.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


PENN_FUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
CLASS_NAMES: Mapping[int, str] = {0: "pedestrian"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP archive without allowing paths outside ``destination``."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def download_pennfudan(workspace: str | Path) -> Path:
    """Download and extract Penn-Fudan once, returning its source directory."""
    workspace = _as_path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    source_root = workspace / "PennFudanPed"
    if (source_root / "PNGImages").is_dir() and (source_root / "PedMasks").is_dir():
        return source_root

    archive_path = workspace / "PennFudanPed.zip"
    if archive_path.exists() and not zipfile.is_zipfile(archive_path):
        archive_path.unlink()
    if not archive_path.exists():
        print("Downloading the Penn-Fudan Pedestrian dataset...")
        partial_path = workspace / "PennFudanPed.zip.part"
        partial_path.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(PENN_FUDAN_URL, timeout=60) as response, partial_path.open("wb") as output:
                expected_size = response.headers.get("Content-Length")
                bytes_written = 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    bytes_written += len(chunk)
            if expected_size is not None and bytes_written != int(expected_size):
                raise RuntimeError(f"Download ended at {bytes_written} bytes; expected {expected_size}")
            if not zipfile.is_zipfile(partial_path):
                raise RuntimeError("The downloaded file is not a valid ZIP archive")
            partial_path.replace(archive_path)
        finally:
            partial_path.unlink(missing_ok=True)
    print("Extracting the dataset...")
    _safe_extract(archive_path, workspace)
    if not (source_root / "PNGImages").is_dir() or not (source_root / "PedMasks").is_dir():
        raise RuntimeError("Penn-Fudan extraction did not contain PNGImages and PedMasks")
    return source_root


def boxes_from_instance_mask(mask_path: str | Path) -> np.ndarray:
    """Return pixel-space ``[x_min, y_min, x_max, y_max]`` boxes from a mask."""
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a single-channel instance mask, received {mask.shape}")
    boxes: list[list[float]] = []
    for object_id in np.unique(mask):
        if object_id == 0:
            continue
        rows, columns = np.where(mask == object_id)
        if len(rows) == 0:
            continue
        boxes.append(
            [
                float(columns.min()),
                float(rows.min()),
                float(columns.max() + 1),
                float(rows.max() + 1),
            ]
        )
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def xyxy_to_yolo(boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Convert pixel-space boxes to normalized YOLO box coordinates."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=float)
    widths_heights = boxes[:, 2:] - boxes[:, :2]
    centers = boxes[:, :2] + widths_heights / 2.0
    result = np.column_stack(
        (
            centers[:, 0] / image_width,
            centers[:, 1] / image_height,
            widths_heights[:, 0] / image_width,
            widths_heights[:, 1] / image_height,
        )
    )
    return np.clip(result, 0.0, 1.0)


def read_yolo_labels(label_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read class IDs and normalized YOLO boxes from one label file."""
    label_path = Path(label_path)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.empty((0,), dtype=int), np.empty((0, 4), dtype=float)
    rows: list[list[float]] = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain 5 values")
        rows.append([float(value) for value in parts])
    array = np.asarray(rows, dtype=float)
    return array[:, 0].astype(int), array[:, 1:]


def write_yolo_labels(label_path: str | Path, class_ids: Iterable[int], boxes: np.ndarray) -> None:
    """Write normalized YOLO labels for one image."""
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    class_ids = np.asarray(list(class_ids), dtype=int)
    if len(class_ids) != len(boxes):
        raise ValueError("class_ids and boxes must contain the same number of entries")
    lines = [
        f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
        for class_id, (x_center, y_center, width, height) in zip(class_ids, boxes)
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _split_names(names: Sequence[str], seed: int) -> dict[str, list[str]]:
    if len(names) < 10:
        raise ValueError("At least 10 images are required to create train, validation, and test splits")
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = round(count * 0.70)
    validation_end = train_end + round(count * 0.15)
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:validation_end]),
        "test": sorted(shuffled[validation_end:]),
    }


def write_dataset_yaml(dataset_root: str | Path) -> Path:
    """Write the portable Ultralytics data configuration for one-class detection."""
    dataset_root = _as_path(dataset_root)
    yaml_path = dataset_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "names:",
                "  0: pedestrian",
                "",
            ]
        )
    )
    return yaml_path


def prepare_base_dataset(
    source_root: str | Path,
    dataset_root: str | Path,
    seed: int = 2026,
    overwrite: bool = False,
) -> Path:
    """Convert Penn-Fudan masks into a fixed YOLO train/val/test dataset."""
    source_root = _as_path(source_root)
    dataset_root = _as_path(dataset_root)
    image_dir = source_root / "PNGImages"
    mask_dir = source_root / "PedMasks"
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() == ".png")
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")
    names = [path.stem for path in image_paths]
    missing_masks = [name for name in names if not (mask_dir / f"{name}_mask.png").exists()]
    if missing_masks:
        raise FileNotFoundError(f"Missing masks for {len(missing_masks)} images")

    if dataset_root.exists() and any(dataset_root.iterdir()):
        if not overwrite:
            data_yaml = dataset_root / "data.yaml"
            if data_yaml.exists():
                return dataset_root
            raise FileExistsError(f"{dataset_root} already exists; choose a new directory or set overwrite=True")
        shutil.rmtree(dataset_root)

    split_names = _split_names(names, seed)
    (dataset_root / "splits_2026.json").parent.mkdir(parents=True, exist_ok=True)
    (dataset_root / "splits_2026.json").write_text(json.dumps(split_names, indent=2) + "\n")
    for split in split_names:
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)

    image_by_name = {path.stem: path for path in image_paths}
    for split, split_image_names in split_names.items():
        for name in split_image_names:
            image_path = image_by_name[name]
            mask_path = mask_dir / f"{name}_mask.png"
            with Image.open(image_path) as image:
                width, height = image.size
            boxes = xyxy_to_yolo(boxes_from_instance_mask(mask_path), width, height)
            destination_image = dataset_root / split / "images" / image_path.name
            destination_label = dataset_root / split / "labels" / f"{name}.txt"
            shutil.copy2(image_path, destination_image)
            write_yolo_labels(destination_label, np.zeros(len(boxes), dtype=int), boxes)

    write_dataset_yaml(dataset_root)
    validate_yolo_dataset(dataset_root)
    return dataset_root


def dataset_summary(dataset_root: str | Path) -> dict[str, dict[str, int]]:
    """Count images and boxes for each split in a YOLO-format dataset."""
    dataset_root = _as_path(dataset_root)
    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        images = sorted(
            path
            for path in (dataset_root / split / "images").iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        boxes = 0
        for image_path in images:
            _, image_boxes = read_yolo_labels(dataset_root / split / "labels" / f"{image_path.stem}.txt")
            boxes += len(image_boxes)
        summary[split] = {"images": len(images), "boxes": boxes}
    return summary


def validate_yolo_dataset(dataset_root: str | Path) -> dict[str, dict[str, int]]:
    """Validate image/label pairs and normalized labels, then return a summary."""
    dataset_root = _as_path(dataset_root)
    errors: list[str] = []
    for split in ("train", "val", "test"):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"Missing image or label directory for {split}")
            continue
        image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not image_paths:
            errors.append(f"No images found in {split}")
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                errors.append(f"Missing label file: {label_path}")
                continue
            try:
                class_ids, boxes = read_yolo_labels(label_path)
            except ValueError as error:
                errors.append(str(error))
                continue
            if np.any(class_ids != 0):
                errors.append(f"Unexpected class ID in {label_path}")
            if len(boxes) and (np.any(boxes < 0.0) or np.any(boxes > 1.0)):
                errors.append(f"Coordinates outside [0, 1] in {label_path}")
            if len(boxes) and (np.any(boxes[:, 2:] <= 0.0)):
                errors.append(f"Non-positive box width or height in {label_path}")
            if len(boxes) and (np.any(xywh_to_xyxy(boxes) < 0.0) or np.any(xywh_to_xyxy(boxes) > 1.0)):
                errors.append(f"A box extends outside the image in {label_path}")
    if errors:
        message = "\n".join(f"- {error}" for error in errors[:20])
        raise ValueError(f"Dataset validation failed:\n{message}")
    return dataset_summary(dataset_root)


def load_yolo_sample(
    dataset_root: str | Path,
    split: str,
    index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Load one image and its normalized boxes for visual inspection."""
    dataset_root = _as_path(dataset_root)
    image_paths = sorted(
        path
        for path in (dataset_root / split / "images").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {dataset_root / split / 'images'}")
    image_path = image_paths[index % len(image_paths)]
    class_ids, boxes = read_yolo_labels(dataset_root / split / "labels" / f"{image_path.stem}.txt")
    with Image.open(image_path) as image:
        image_array = np.asarray(image.convert("RGB"))
    return image_array, class_ids, boxes, image_path


def plot_yolo_sample(axes, image: np.ndarray, class_ids: np.ndarray, boxes: np.ndarray) -> None:
    """Draw normalized YOLO boxes onto a Matplotlib axes."""
    import matplotlib.patches

    axes.imshow(image)
    image_height, image_width = image.shape[:2]
    for class_id, (x_center, y_center, width, height) in zip(class_ids, boxes):
        box_width = width * image_width
        box_height = height * image_height
        x_min = x_center * image_width - box_width / 2.0
        y_min = y_center * image_height - box_height / 2.0
        rectangle = matplotlib.patches.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            fill=False,
            edgecolor="#dc2626",
            linewidth=2,
        )
        axes.add_patch(rectangle)
        axes.text(
            x_min,
            max(12, y_min - 3),
            CLASS_NAMES.get(int(class_id), str(class_id)),
            color="white",
            fontsize=9,
            bbox={"facecolor": "#dc2626", "edgecolor": "none", "pad": 1.5},
        )
    axes.axis("off")


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert normalized ``x_center, y_center, width, height`` boxes to corners."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=float)
    half_size = boxes[:, 2:] / 2.0
    return np.column_stack((boxes[:, :2] - half_size, boxes[:, :2] + half_size))


def pairwise_iou_xywh(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for normalized YOLO-format boxes."""
    corners1 = xywh_to_xyxy(boxes1)
    corners2 = xywh_to_xyxy(boxes2)
    if len(corners1) == 0 or len(corners2) == 0:
        return np.zeros((len(corners1), len(corners2)), dtype=float)
    top_left = np.maximum(corners1[:, None, :2], corners2[None, :, :2])
    bottom_right = np.minimum(corners1[:, None, 2:], corners2[None, :, 2:])
    overlap_size = np.clip(bottom_right - top_left, 0.0, None)
    intersection = overlap_size[..., 0] * overlap_size[..., 1]
    area1 = np.prod(np.clip(corners1[:, 2:] - corners1[:, :2], 0.0, None), axis=1)
    area2 = np.prod(np.clip(corners2[:, 2:] - corners2[:, :2], 0.0, None), axis=1)
    union = area1[:, None] + area2[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def reference_labels_for_image(source_root: str | Path, image_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Derive normalized reference labels for one Penn-Fudan image."""
    source_root = _as_path(source_root)
    image_path = source_root / "PNGImages" / image_name
    mask_path = source_root / "PedMasks" / f"{Path(image_name).stem}_mask.png"
    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Could not find image and mask for {image_name}")
    with Image.open(image_path) as image:
        width, height = image.size
    boxes = xyxy_to_yolo(boxes_from_instance_mask(mask_path), width, height)
    return np.zeros(len(boxes), dtype=int), boxes


def annotation_quality_report(
    source_root: str | Path,
    image_name: str,
    manual_label_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Compare manually created labels with mask-derived reference boxes."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie between 0 and 1")
    manual_class_ids, manual_boxes = read_yolo_labels(manual_label_path)
    if np.any(manual_class_ids != 0):
        raise ValueError("This lab uses only class ID 0 for pedestrian")
    if len(manual_boxes) and (
        np.any(manual_boxes < 0.0)
        or np.any(manual_boxes > 1.0)
        or np.any(manual_boxes[:, 2:] <= 0.0)
        or np.any(xywh_to_xyxy(manual_boxes) < 0.0)
        or np.any(xywh_to_xyxy(manual_boxes) > 1.0)
    ):
        raise ValueError("Manual YOLO boxes must stay inside the image with positive width and height")

    reference_class_ids, reference_boxes = reference_labels_for_image(source_root, image_name)
    overlaps = pairwise_iou_xywh(manual_boxes, reference_boxes)
    matched_reference: set[int] = set()
    matched_pairs: list[tuple[int, int, float]] = []
    extra_indices: list[int] = []
    duplicate_indices: list[int] = []
    for manual_index, row in enumerate(overlaps):
        available = np.ones(len(reference_boxes), dtype=bool)
        if matched_reference:
            available[list(matched_reference)] = False
        if np.any(available):
            available_ious = np.where(available, row, -np.inf)
            reference_index = int(np.argmax(available_ious))
            if available_ious[reference_index] >= iou_threshold:
                matched_reference.add(reference_index)
                matched_pairs.append((manual_index, reference_index, float(available_ious[reference_index])))
                continue
        if len(reference_boxes) and float(np.max(row)) >= iou_threshold:
            duplicate_indices.append(manual_index)
        else:
            extra_indices.append(manual_index)

    missed_indices = sorted(set(range(len(reference_boxes))) - matched_reference)
    return {
        "manual_class_ids": manual_class_ids,
        "manual_boxes": manual_boxes,
        "reference_class_ids": reference_class_ids,
        "reference_boxes": reference_boxes,
        "matches": matched_pairs,
        "extra_indices": extra_indices,
        "duplicate_indices": duplicate_indices,
        "missed_indices": missed_indices,
    }


def make_labeling_practice_bundle(
    source_root: str | Path,
    output_root: str | Path,
    image_count: int = 3,
    overwrite: bool = False,
) -> Path:
    """Create a small unlabeled image bundle for browser-based annotation."""
    source_root = _as_path(source_root)
    output_root = _as_path(output_root)
    image_paths = sorted((source_root / "PNGImages").glob("*.png"))
    if image_count < 1 or image_count > len(image_paths):
        raise ValueError("image_count must select at least one available image")
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            return output_root / "pennfudan_labeling_images.zip"
        shutil.rmtree(output_root)
    images_output = output_root / "images"
    images_output.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths[:image_count]:
        shutil.copy2(image_path, images_output / image_path.name)
    archive_base = output_root / "pennfudan_labeling_images"
    shutil.make_archive(str(archive_base), "zip", images_output)
    return archive_base.with_suffix(".zip")


def make_augmented_dataset(
    base_dataset_root: str | Path,
    augmented_dataset_root: str | Path,
    copies_per_image: int = 1,
    seed: int = 2026,
    overwrite: bool = False,
) -> Path:
    """Create a separate YOLO dataset with augmented copies of train images only."""
    try:
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        import albumentations as A
    except ImportError as error:
        raise ImportError("Install albumentations before creating augmented images") from error

    base_dataset_root = _as_path(base_dataset_root)
    augmented_dataset_root = _as_path(augmented_dataset_root)
    if copies_per_image < 1:
        raise ValueError("copies_per_image must be at least 1")
    validate_yolo_dataset(base_dataset_root)
    if augmented_dataset_root.exists() and any(augmented_dataset_root.iterdir()):
        if not overwrite:
            if (augmented_dataset_root / "data.yaml").exists():
                return augmented_dataset_root
            raise FileExistsError(
                f"{augmented_dataset_root} already exists; choose a new directory or set overwrite=True"
            )
        shutil.rmtree(augmented_dataset_root)
    shutil.copytree(base_dataset_root, augmented_dataset_root)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.4),
    )
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)

    train_image_dir = base_dataset_root / "train" / "images"
    train_label_dir = base_dataset_root / "train" / "labels"
    output_image_dir = augmented_dataset_root / "train" / "images"
    output_label_dir = augmented_dataset_root / "train" / "labels"
    image_paths = sorted(path for path in train_image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in image_paths:
        class_ids, boxes = read_yolo_labels(train_label_dir / f"{image_path.stem}.txt")
        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("RGB"))
        for copy_index in range(copies_per_image):
            result = transform(
                image=image_array,
                bboxes=boxes.tolist(),
                class_labels=class_ids.tolist(),
            )
            output_name = f"{image_path.stem}_aug{copy_index + 1}.png"
            Image.fromarray(result["image"]).save(output_image_dir / output_name)
            write_yolo_labels(
                output_label_dir / f"{Path(output_name).stem}.txt",
                result["class_labels"],
                np.asarray(result["bboxes"], dtype=float).reshape(-1, 4),
            )

    write_dataset_yaml(augmented_dataset_root)
    validate_yolo_dataset(augmented_dataset_root)
    return augmented_dataset_root
