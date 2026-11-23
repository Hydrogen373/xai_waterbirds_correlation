"""Reproducible CUB/Places composites for three Waterbirds correlations."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CORRELATIONS = (0.95, 0.80, 0.65)
WATER_SPECIES = (
    "albatross", "auklet", "cormorant", "frigatebird", "fulmar", "gull",
    "jaeger", "kittiwake", "pelican", "puffin", "tern", "gadwall",
    "grebe", "mallard", "merganser", "guillemot", "pacific_loon",
)
PLACE_DIRS = {
    0: ("b/bamboo_forest", "f/forest/broadleaf"),
    1: ("o/ocean", "l/lake/natural"),
}
FIELDS = ("image", "bird_id", "bird_path", "background_path", "label", "place", "split")


def correlation_name(correlation: float) -> str:
    return f"corr{round(correlation * 100):03d}"


def _read_id_table(path: Path) -> dict[int, str]:
    return {int(parts[0]): parts[1] for line in path.read_text(encoding="utf-8").splitlines()
            if (parts := line.split())}


def _bird_label(relative_path: str) -> int:
    species = relative_path.split("/")[0].split(".", 1)[1].lower()
    return int(any(name in species for name in WATER_SPECIES))


def _split_birds(cub_root: Path, seed: int) -> dict[str, list[tuple[int, str, int]]]:
    images = _read_id_table(cub_root / "images.txt")
    original_split = _read_id_table(cub_root / "train_test_split.txt")
    if images.keys() != original_split.keys():
        raise ValueError("CUB image and split IDs do not match")
    groups = defaultdict(list)
    for bird_id, relative_path in images.items():
        if original_split[bird_id] == "1":
            groups[_bird_label(relative_path)].append((bird_id, relative_path, _bird_label(relative_path)))
    if set(groups) != {0, 1}:
        raise ValueError("CUB training split must contain both bird classes")
    rng = np.random.default_rng(seed)
    result = {"train": [], "validation": []}
    for label in (0, 1):
        records = sorted(groups[label])
        rng.shuffle(records)
        n_validation = round(len(records) * 0.2)
        result["validation"].extend(records[:n_validation])
        result["train"].extend(records[n_validation:])
    for split in result:
        result[split].sort()
    return result


def _place_pools(places_root: Path, required: int, seed: int) -> dict[int, list[Path]]:
    rng = np.random.default_rng(seed + 1)
    pools = {}
    for place, categories in PLACE_DIRS.items():
        paths = sorted(path for category in categories
                       for path in (places_root / "data_256" / category).glob("*.jpg"))
        if len(paths) < required:
            raise ValueError(f"Need {required} background images for place={place}; found {len(paths)}")
        rng.shuffle(paths)
        pools[place] = paths
    return pools


def _assignment(split_birds: dict[str, list[tuple[int, str, int]]],
                place_pools: dict[int, list[Path]], seed: int) -> dict:
    """Choose each bird's two backgrounds once; place assignments vary by p."""
    rng = np.random.default_rng(seed + 2)
    assignments = {}
    offset = 0
    for split in ("train", "validation"):
        birds = split_birds[split]
        for position, (bird_id, _, _) in enumerate(birds):
            assignments[bird_id] = {
                0: place_pools[0][offset + position],
                1: place_pools[1][offset + position],
            }
        offset += len(birds)
    orders = {}
    for split in ("train", "validation"):
        for label in (0, 1):
            ids = sorted(bird_id for bird_id, _, y in split_birds[split] if y == label)
            rng.shuffle(ids)
            orders[(split, label)] = ids
    return assignments, orders


def _compose(bird_path: Path, mask_path: Path, background_path: Path, output: Path) -> None:
    with Image.open(bird_path) as source, Image.open(mask_path) as mask, Image.open(background_path) as bg:
        source = source.convert("RGB")
        mask = mask.convert("L")
        if mask.size != source.size:
            raise ValueError(f"Mask size differs from bird image: {mask_path}")
        # ImageOps.fit center-crops the background to the original bird dimensions.
        from PIL import ImageOps
        background = ImageOps.fit(bg.convert("RGB"), source.size, method=Image.Resampling.LANCZOS)
        combined = Image.composite(source, background, mask)
        output.parent.mkdir(parents=True, exist_ok=True)
        combined.save(output, quality=95)


def prepare_synthetic_data(cub_root: Path, places_root: Path, output_dir: Path,
                           seed: int) -> dict[float, dict[str, list[dict]]]:
    """Write one composite per bird, split and correlation; reuse existing manifests."""
    cub_root = cub_root.resolve()
    places_root = places_root.resolve()
    data_dir = output_dir / "data"
    settings_path = data_dir / "settings.json"
    expected_settings = {"cub_root": str(cub_root), "places_root": str(places_root),
                         "seed": seed, "correlations": list(CORRELATIONS)}
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != expected_settings:
            raise ValueError("Existing synthetic data were prepared with different settings")
        result = {}
        for corr in CORRELATIONS:
            name = correlation_name(corr)
            result[corr] = {}
            for split in ("train", "validation"):
                with (data_dir / name / f"{split}.csv").open(newline="", encoding="utf-8") as file:
                    rows = list(csv.DictReader(file))
                for row in rows:
                    for field in ("bird_id", "label", "place"):
                        row[field] = int(row[field])
                if not rows or any(not (output_dir / row["image"]).is_file() for row in rows):
                    raise ValueError(f"Synthetic images missing for {name}/{split}")
                result[corr][split] = rows
        return result

    split_birds = _split_birds(cub_root, seed)
    count = sum(map(len, split_birds.values()))
    pools = _place_pools(places_root, count, seed)
    backgrounds, orders = _assignment(split_birds, pools, seed)
    result = {}
    for corr in CORRELATIONS:
        name = correlation_name(corr)
        result[corr] = {}
        for split in ("train", "validation"):
            aligned = set()
            for label in (0, 1):
                ids = orders[(split, label)]
                aligned.update(ids[:round(corr * len(ids))])
            rows = []
            for bird_id, relative_path, label in split_birds[split]:
                place = label if bird_id in aligned else 1 - label
                bird_path = cub_root / "images" / relative_path
                mask_path = cub_root / "segmentations" / Path(relative_path).with_suffix(".png")
                bg_path = backgrounds[bird_id][place]
                relative_output = Path("data") / name / split / f"{bird_id:05d}.jpg"
                _compose(bird_path, mask_path, bg_path, output_dir / relative_output)
                rows.append({"image": relative_output.as_posix(), "bird_id": bird_id,
                             "bird_path": str(bird_path), "background_path": str(bg_path),
                             "label": label, "place": place, "split": split})
            manifest = data_dir / name / f"{split}.csv"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            result[corr][split] = rows
    data_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(expected_settings, indent=2), encoding="utf-8")
    counts = {}
    for corr, splits in result.items():
        counts[correlation_name(corr)] = {}
        for split, rows in splits.items():
            groups = Counter((int(r["label"]), int(r["place"])) for r in rows)
            counts[correlation_name(corr)][split] = {
                "groups": {f"{y}{p}": groups[(y, p)] for y in (0, 1) for p in (0, 1)},
                "realized_correlation_by_label": {
                    str(y): groups[(y, y)] / (groups[(y, 0)] + groups[(y, 1)])
                    for y in (0, 1)
                },
            }
    (data_dir / "group_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return result
