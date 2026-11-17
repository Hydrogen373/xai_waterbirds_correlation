"""Small, offline checks for the experiment's most failure-prone behavior."""

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

from experiment import EpochEvaluator, GradCAMVisualizer, Trainer, WaterbirdsData, WaterbirdsImages


class FakeRows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def select(self, indices):
        return FakeRows([self.rows[index] for index in indices])


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

    @property
    def last_convolution(self):
        return self.conv


class ExperimentTests(unittest.TestCase):
    def test_split_is_disjoint_and_stratified(self):
        rows = FakeRows(
            [{"image": Image.new("RGB", (32, 32)), "label": label, "place": place}
             for label in (0, 1) for place in (0, 1) for _ in range(10)]
        )
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda _name, split: rows
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            data = WaterbirdsData(batch_size=4, seed=42)
        train = set(data.split_indices["train"])
        validation = set(data.split_indices["validation"])
        self.assertEqual(len(train), 32)
        self.assertEqual(len(validation), 8)
        self.assertFalse(train & validation)
        self.assertEqual(
            {(label, place): sum(row["label"] == label and row["place"] == place
                                 for row in data.validation.rows.rows)
             for label in (0, 1) for place in (0, 1)},
            {(label, place): 2 for label in (0, 1) for place in (0, 1)},
        )

    def test_group_accuracy_exposes_background_shortcut(self):
        places = torch.tensor([0, 1, 0, 1])
        labels = torch.tensor([0, 0, 1, 1])
        images = places.float().reshape(4, 1, 1, 1)
        loader = DataLoader(TensorDataset(images, labels, places), batch_size=4)
        metrics = EpochEvaluator(torch.device("cpu")).evaluate(PlaceClassifier(), loader)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["aligned_accuracy"], 1.0)
        self.assertEqual(metrics["conflicting_accuracy"], 0.0)
        self.assertEqual(metrics["accuracy_gap"], 1.0)

    def test_train_epoch_updates_weights(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(3, 2))
        before = model[1].weight.detach().clone()
        images = torch.randn(4, 3, 1, 1)
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(images, labels, labels), batch_size=2)
        metrics = Trainer(model, torch.device("cpu"), epochs=2).train_epoch(loader)
        self.assertTrue(np.isfinite(metrics["train_loss"]))
        self.assertFalse(torch.equal(before, model[1].weight))

    def test_gradcam_writes_fixed_sample_panels(self):
        image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        rows = FakeRows([{"image": image, "label": label, "place": label}
                         for label in (0, 1)])
        dataset = WaterbirdsImages(rows)
        with tempfile.TemporaryDirectory() as directory:
            visualizer = GradCAMVisualizer(
                TinyCAMModel(), dataset, [0, 1], torch.device("cpu"), Path(directory)
            )
            visualizer.render(0)
            visualizer.render(5)
            visualizer.save_timeline()
            self.assertTrue((Path(directory) / "gradcam_epoch_000.png").exists())
            self.assertTrue((Path(directory) / "gradcam_timeline.png").exists())
            self.assertEqual(
                len((Path(directory) / "gradcam_changes.csv").read_text().splitlines()),
                5,
            )


if __name__ == "__main__":
    unittest.main()
