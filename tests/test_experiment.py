"""Offline checks for data composition, metrics, and the GradCAM layout."""

import argparse
import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment import (EpochEvaluator, GradCAMVisualizer, Trainer, WaterbirdsImages,
                        build_gradcam_grid, run)
from waterbirds_data import CORRELATIONS, correlation_name, prepare_synthetic_data


class PlaceClassifier(nn.Module):
    def forward(self, images):
        place = images[:, 0, 0, 0]
        return torch.stack((1 - place, place), dim=1)


class TinyCAMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 3, padding=1)
        self.head = nn.Linear(2, 2)

    def forward(self, images):
        features = torch.relu(self.conv(images))
        return self.head(features.mean(dim=(2, 3)))

    def forward_with_features(self, images):
        features = torch.relu(self.conv(images)).mean(dim=(2, 3))
        return self.head(features), features

    @property
    def last_convolution(self):
        return self.conv


def make_fixture(root: Path):
    cub = root / "CUB_200_2011"
    places = root / "places365"
    image_lines, split_lines = [], []
    for label, species in ((0, "Cardinal"), (1, "Albatross")):
        for number in range(10):
            bird_id = label * 10 + number + 1
            relative = f"{label + 1:03d}.{species}/{number:03d}.jpg"
            image_lines.append(f"{bird_id} {relative}")
            split_lines.append(f"{bird_id} 1")
            image_path = cub / "images" / relative
            mask_path = cub / "segmentations" / Path(relative).with_suffix(".png")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 12), (255, 0, 0) if label else (0, 0, 255)).save(image_path)
            mask = Image.new("L", (12, 12), 0)
            mask.putpixel((6, 6), 255)
            mask.save(mask_path)
    (cub / "images.txt").write_text("\n".join(image_lines), encoding="utf-8")
    (cub / "train_test_split.txt").write_text("\n".join(split_lines), encoding="utf-8")
    for category, color in (("b/bamboo_forest", (0, 255, 0)),
                            ("f/forest/broadleaf", (0, 200, 0)),
                            ("o/ocean", (0, 255, 255)),
                            ("l/lake/natural", (0, 200, 255))):
        folder = places / "data_256" / category
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(12):
            Image.new("RGB", (12, 12), color).save(folder / f"{index:03d}.jpg")
    return cub, places


class ExperimentTests(unittest.TestCase):
    def test_composites_preserve_birds_and_split_for_all_correlations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cub, places = make_fixture(root)
            output = root / "output"
            manifests = prepare_synthetic_data(cub, places, output, seed=42)
            base_train = {row["bird_id"] for row in manifests[0.95]["train"]}
            base_val = {row["bird_id"] for row in manifests[0.95]["validation"]}
            self.assertEqual(len(base_train), 16)
            self.assertEqual(len(base_val), 4)
            self.assertFalse(base_train & base_val)
            for correlation in CORRELATIONS:
                splits = manifests[correlation]
                self.assertEqual({row["bird_id"] for row in splits["train"]}, base_train)
                self.assertEqual({row["bird_id"] for row in splits["validation"]}, base_val)
                train_bgs = {row["background_path"] for row in splits["train"]}
                val_bgs = {row["background_path"] for row in splits["validation"]}
                self.assertFalse(train_bgs & val_bgs)
                for split, rows in splits.items():
                    self.assertEqual(len({row["bird_id"] for row in rows}), len(rows))
                    for label in (0, 1):
                        matching = [int(row["label"] == row["place"])
                                    for row in rows if row["label"] == label]
                        self.assertEqual(sum(matching), round(correlation * len(matching)))
                    self.assertTrue(all((output / row["image"]).is_file() for row in rows))
            self.assertEqual(prepare_synthetic_data(cub, places, output, seed=42)[0.8]["train"],
                             manifests[0.8]["train"])
            self.assertTrue((output / "data" / "group_counts.json").is_file())

    def test_evaluator_exposes_group_shortcut(self):
        places = torch.tensor([0, 1, 0, 1])
        labels = torch.tensor([0, 0, 1, 1])
        images = places.float().reshape(4, 1, 1, 1)
        loader = DataLoader(TensorDataset(images, labels, places), batch_size=4)
        metrics = EpochEvaluator(torch.device("cpu")).evaluate(PlaceClassifier(), loader)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["group_00_accuracy"], 1.0)
        self.assertEqual(metrics["group_01_accuracy"], 0.0)

    def test_train_epoch_updates_weights(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(3, 2))
        before = model[1].weight.detach().clone()
        images = torch.randn(4, 3, 1, 1)
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(images, labels, labels), batch_size=2)
        metrics = Trainer(model, torch.device("cpu"), epochs=2).train_epoch(loader)
        self.assertTrue(np.isfinite(metrics["train_loss"]))
        self.assertFalse(torch.equal(before, model[1].weight))

    def test_gradcam_grid_has_epoch_rows_and_nine_columns(self):
        image = Image.new("RGB", (32, 32), (128, 128, 128))
        rows = [{"image": image, "label": label, "place": place}
                for label, place in ((1, 1), (1, 0), (0, 1))]
        dataset = WaterbirdsImages(rows)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for correlation in CORRELATIONS:
                visualizer = GradCAMVisualizer(
                    TinyCAMModel(), dataset, {"a": 0, "b": 1, "c": 2},
                    torch.device("cpu"), output / correlation_name(correlation) / "cam")
                visualizer.render(1)
                visualizer.render(2)
            build_gradcam_grid(output, 2)
            with Image.open(output / "gradcam_grid.png") as figure:
                self.assertEqual(figure.size, (70 + 9 * 128, 38 + 2 * 128))

    def test_one_epoch_orchestration_and_resume(self):
        image = Image.new("RGB", (32, 32), (128, 128, 128))
        rows = [{"image": image, "label": y, "place": p}
                for y in (0, 1) for p in (0, 1)]
        manifests = {corr: {"train": rows, "validation": rows} for corr in CORRELATIONS}

        class FakeRows(list):
            def __getitem__(self, key):
                if isinstance(key, str):
                    return [row[key] for row in self]
                return super().__getitem__(key)

        class FakeProbe:
            def __init__(self, _device, _seed):
                pass

            def extract(self, _model, _loader):
                return None

            def evaluate(self, _train, _test):
                return {"bird_probe_accuracy": 0.5, "background_probe_accuracy": 0.5}

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda _name, split: FakeRows(rows)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = argparse.Namespace(output_dir=output, cub_root=Path(directory),
                                      places_root=Path(directory), epochs=1, batch_size=4,
                                      num_workers=0, seed=42, device="cpu", resume=False)
            with patch.dict(sys.modules, {"datasets": fake_datasets}), \
                 patch("experiment.prepare_synthetic_data", return_value=manifests), \
                 patch("experiment.VGG11BirdClassifier", TinyCAMModel), \
                 patch("experiment.LinearProbeEvaluator", FakeProbe):
                run(args)
                args.resume = True
                run(args)
            with (output / "epoch_metrics.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 3)
            self.assertTrue((output / "epoch_curves.png").is_file())
            self.assertTrue((output / "gradcam_grid.png").is_file())
            self.assertTrue(all((output / correlation_name(c) / "final.pt").is_file()
                                for c in CORRELATIONS))


if __name__ == "__main__":
    unittest.main()
