"""Track shortcut information and classification behavior while training VGG11."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import vgg11


class WaterbirdsImages(Dataset):
    """Expose Waterbirds rows as images and their two binary attributes."""

    def __init__(self, rows, training: bool = False):
        self.rows = rows
        self.display_transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
        image_transforms = [self.display_transform]
        if training:
            image_transforms.append(T.RandomHorizontalFlip())
        image_transforms.extend(
            [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)]
        )
        self.transform = T.Compose(image_transforms)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = row["image"].convert("RGB")
        return self.transform(image), int(row["label"]), int(row["place"])

    def display_image(self, index: int) -> Image.Image:
        """Return the exact crop shown to the model, without normalization."""
        return self.display_transform(self.rows[index]["image"].convert("RGB"))


class WaterbirdsData:
    """Own the fixed group-stratified split and its data loaders."""

    def __init__(self, batch_size: int, seed: int, num_workers: int = 0):
        from datasets import load_dataset

        source = load_dataset("grodino/waterbirds", split="train")
        groups = defaultdict(list)
        for index, (label, place) in enumerate(zip(source["label"], source["place"])):
            groups[(int(label), int(place))].append(index)
        if len(groups) != 4:
            raise ValueError("Expected all four label/place groups in the train split")

        rng = np.random.default_rng(seed)
        train_indices, validation_indices = [], []
        for group in sorted(groups):
            indices = np.asarray(groups[group], dtype=np.int64)
            rng.shuffle(indices)
            validation_count = max(1, round(len(indices) * 0.2))
            validation_indices.extend(indices[:validation_count].tolist())
            train_indices.extend(indices[validation_count:].tolist())
        rng.shuffle(train_indices)
        rng.shuffle(validation_indices)

        self.split_indices = {
            "train": train_indices,
            "validation": validation_indices,
        }
        self.train = WaterbirdsImages(source.select(train_indices), training=True)
        self.train_for_probe = WaterbirdsImages(source.select(train_indices))
        self.validation = WaterbirdsImages(source.select(validation_indices))
        self.batch_size = batch_size
        self.num_workers = num_workers

    def loader(self, dataset: WaterbirdsImages, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_loader(self) -> DataLoader:
        """Keep the official test split untouched until final evaluation."""
        from datasets import load_dataset

        test = WaterbirdsImages(load_dataset("grodino/waterbirds", split="test"))
        return self.loader(test)


class VGG11BirdClassifier(nn.Module):
    """Own the randomly initialized two-class VGG11 architecture."""

    def __init__(self):
        super().__init__()
        self.network = vgg11(weights=None)
        self.network.classifier[6] = nn.Linear(4096, 2)
        # A non-in-place ReLU keeps the last convolution output usable by Grad-CAM.
        self.network.features[19].inplace = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def hidden_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the last hidden activation, before dropout and final logits."""
        features = self.network.features(images)
        features = self.network.avgpool(features).flatten(1)
        return self.network.classifier[:5](features)

    @property
    def last_convolution(self) -> nn.Module:
        return self.network.features[18]


class Trainer:
    """Update model weights for one epoch at a time."""

    def __init__(self, model: nn.Module, device: torch.device, epochs: int):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        learning_rate = self.optimizer.param_groups[0]["lr"]
        total_loss, sample_count = 0.0, 0
        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * labels.numel()
            sample_count += labels.numel()
        self.scheduler.step()
        return {"train_loss": total_loss / sample_count, "learning_rate": learning_rate}


class EpochEvaluator:
    """Compute classification accuracy overall and for each Waterbirds group."""

    def __init__(self, device: torch.device):
        self.device = device

    def evaluate(self, model: nn.Module, loader: DataLoader) -> dict[str, float | int]:
        model.eval()
        correct = defaultdict(int)
        counts = defaultdict(int)
        with torch.inference_mode():
            for images, labels, places in loader:
                predictions = model(images.to(self.device)).argmax(dim=1).cpu()
                for label, place, prediction in zip(labels, places, predictions):
                    group = (int(label), int(place))
                    counts[group] += 1
                    correct[group] += int(prediction == label)

        if not counts:
            raise ValueError("Cannot evaluate an empty dataset")
        result: dict[str, float | int] = {}
        group_accuracies = []
        for label in (0, 1):
            for place in (0, 1):
                group = (label, place)
                name = f"group_{label}{place}"
                result[f"{name}_count"] = counts[group]
                result[f"{name}_accuracy"] = (
                    correct[group] / counts[group] if counts[group] else float("nan")
                )
                if counts[group]:
                    group_accuracies.append(correct[group] / counts[group])

        aligned = [(0, 0), (1, 1)]
        conflicting = [(0, 1), (1, 0)]
        aligned_count = sum(counts[group] for group in aligned)
        conflicting_count = sum(counts[group] for group in conflicting)
        result["accuracy"] = sum(correct.values()) / sum(counts.values())
        result["worst_group_accuracy"] = min(group_accuracies)
        result["aligned_accuracy"] = sum(correct[group] for group in aligned) / aligned_count
        result["conflicting_accuracy"] = (
            sum(correct[group] for group in conflicting) / conflicting_count
        )
        result["accuracy_gap"] = result["aligned_accuracy"] - result["conflicting_accuracy"]
        return result


class LinearProbeEvaluator:
    """Measure how linearly readable bird and background labels are."""

    def __init__(self, device: torch.device, seed: int):
        self.device = device
        self.seed = seed

    def _extract(self, model: VGG11BirdClassifier, loader: DataLoader):
        model.eval()
        features, labels, places = [], [], []
        with torch.inference_mode():
            for images, batch_labels, batch_places in loader:
                features.append(model.hidden_features(images.to(self.device)).cpu().numpy())
                labels.append(batch_labels.numpy())
                places.append(batch_places.numpy())
        return np.concatenate(features), np.concatenate(labels), np.concatenate(places)

    def evaluate(
        self,
        model: VGG11BirdClassifier,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> dict[str, float]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        train_features, train_labels, train_places = self._extract(model, train_loader)
        val_features, val_labels, val_places = self._extract(model, validation_loader)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)

        # Equal group mass removes the training split's label/place correlation
        # from the probe's optimization objective.
        train_groups = 2 * train_labels + train_places
        val_groups = 2 * val_labels + val_places
        group_counts = np.bincount(train_groups, minlength=4)
        if np.any(group_counts == 0) or np.any(np.bincount(val_groups, minlength=4) == 0):
            raise ValueError("Linear probes require all four groups in both splits")
        sample_weights = len(train_groups) / (4 * group_counts[train_groups])

        results = {}
        for name, train_target, val_target in (
            ("label", train_labels, val_labels),
            ("place", train_places, val_places),
        ):
            probe = LogisticRegression(C=1.0, max_iter=1000, random_state=self.seed)
            probe.fit(train_features, train_target, sample_weight=sample_weights)
            predictions = probe.predict(val_features)
            results[f"probe_{name}_accuracy"] = float(np.mean(predictions == val_target))
            results[f"probe_{name}_group_mean_accuracy"] = float(
                np.mean(
                    [np.mean(predictions[val_groups == group] == val_target[val_groups == group])
                     for group in range(4)]
                )
            )
        return results


class GradCAMVisualizer:
    """Track Grad-CAM map changes and render ten fixed validation examples."""

    def __init__(
        self,
        model: VGG11BirdClassifier,
        dataset: WaterbirdsImages,
        indices: list[int],
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model
        self.dataset = dataset
        self.indices = indices
        self.device = device
        self.output_dir = output_dir
        self.history: dict[int, list[np.ndarray]] = {}
        self.heatmap_history: dict[int, list[np.ndarray]] = {}

    def render(self, epoch: int) -> None:
        self.model.eval()
        activations = []

        # A forward hook is required to retain the convolution output for autograd.
        def retain_activation(_module, _inputs, output):
            activations.append(output)

        hook = self.model.last_convolution.register_forward_hook(retain_activation)
        overlays = []
        heatmaps = []
        try:
            for index in self.indices:
                activations.clear()
                image, label, _ = self.dataset[index]
                image = image.unsqueeze(0).to(self.device)
                with torch.enable_grad():
                    logits = self.model(image)
                    gradient = torch.autograd.grad(logits[0, label], activations[0])[0]
                    weights = gradient.mean(dim=(2, 3), keepdim=True)
                    heatmap = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
                    heatmap = torch.nn.functional.interpolate(
                        heatmap, size=(224, 224), mode="bilinear", align_corners=False
                    )[0, 0]
                heatmap = heatmap.detach().cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-8)
                heatmaps.append(heatmap)
                original = np.asarray(self.dataset.display_image(index), dtype=np.float32) / 255
                color = matplotlib.colormaps["jet"](heatmap)[..., :3]
                overlays.append(((0.55 * original + 0.45 * color) * 255).astype(np.uint8))
        finally:
            hook.remove()

        previous_epoch = max(self.heatmap_history) if self.heatmap_history else None
        with (self.output_dir / "gradcam_changes.csv").open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            if previous_epoch is None:
                writer.writerow(["epoch", "previous_epoch", "image_index", "mean_absolute_change"])
            for position, index in enumerate(self.indices):
                change = ""
                if previous_epoch is not None:
                    change = float(np.mean(np.abs(
                        heatmaps[position] - self.heatmap_history[previous_epoch][position]
                    )))
                writer.writerow([epoch, previous_epoch if previous_epoch is not None else "", index, change])
        self.heatmap_history[epoch] = heatmaps
        self.history[epoch] = overlays
        figure, axes = plt.subplots(len(self.indices), 2, figsize=(6, 2.5 * len(self.indices)))
        for row, (index, overlay) in enumerate(zip(self.indices, overlays)):
            _, label, place = self.dataset[index]
            axes[row, 0].imshow(self.dataset.display_image(index))
            axes[row, 0].set_ylabel(f"image {index}\ny={label}, place={place}")
            axes[row, 1].imshow(overlay)
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
        axes[0, 0].set_title("Model input")
        axes[0, 1].set_title(f"Grad-CAM · epoch {epoch}")
        figure.tight_layout()
        figure.savefig(self.output_dir / f"gradcam_epoch_{epoch:03d}.png", dpi=120)
        plt.close(figure)

    def save_timeline(self) -> None:
        epochs = sorted(self.history)
        figure, axes = plt.subplots(
            len(self.indices), len(epochs), figsize=(2.3 * len(epochs), 2 * len(self.indices)),
            squeeze=False,
        )
        for row in range(len(self.indices)):
            for column, epoch in enumerate(epochs):
                axis = axes[row, column]
                axis.imshow(self.history[epoch][row])
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"epoch {epoch}")
                if column == 0:
                    axis.set_ylabel(f"image {self.indices[row]}")
        figure.tight_layout()
        figure.savefig(self.output_dir / "gradcam_timeline.png", dpi=120)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vgg11_scratch"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch-size must be positive; num-workers cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error(f"output directory must be empty: {args.output_dir}")

    # Keep the train/validation split and Grad-CAM examples fixed across epochs.
    data = WaterbirdsData(args.batch_size, args.seed, args.num_workers)
    (args.output_dir / "split_indices.json").write_text(
        json.dumps(data.split_indices, indent=2), encoding="utf-8"
    )
    sample_groups = defaultdict(list)
    for index, (label, place) in enumerate(
        zip(data.validation.rows["label"], data.validation.rows["place"])
    ):
        sample_groups[(int(label), int(place))].append(index)
    rng = np.random.default_rng(args.seed)
    sample_indices = []
    for group, amount in zip(sorted(sample_groups), (3, 3, 2, 2)):
        if len(sample_groups[group]) < amount:
            raise ValueError(f"Not enough validation samples in group {group}")
        sample_indices.extend(rng.choice(sample_groups[group], amount, replace=False).tolist())
    (args.output_dir / "gradcam_indices.json").write_text(
        json.dumps(sample_indices, indent=2), encoding="utf-8"
    )
    (args.output_dir / "settings.json").write_text(
        json.dumps({"dataset": "grodino/waterbirds", "weights": None,
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "num_workers": args.num_workers, "seed": args.seed,
                    "device": str(device), "probe_C": 1.0}, indent=2), encoding="utf-8"
    )

    model = VGG11BirdClassifier().to(device)
    trainer = Trainer(model, device, args.epochs)
    evaluator = EpochEvaluator(device)
    probe_evaluator = LinearProbeEvaluator(device, args.seed)
    visualizer = GradCAMVisualizer(model, data.validation, sample_indices, device, args.output_dir)
    train_loader = data.loader(data.train, shuffle=True)
    probe_train_loader = data.loader(data.train_for_probe)
    validation_loader = data.loader(data.validation)

    torch.save({"epoch": 0, "model": model.state_dict()}, args.output_dir / "checkpoint_000.pt")
    visualizer.render(0)
    with (args.output_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for epoch in range(1, args.epochs + 1):
            metrics = {"epoch": epoch}
            metrics.update(trainer.train_epoch(train_loader))
            metrics.update(evaluator.evaluate(model, validation_loader))
            metrics.update(probe_evaluator.evaluate(model, probe_train_loader, validation_loader))
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(metrics))
                writer.writeheader()
            writer.writerow(metrics)
            file.flush()
            print(f"epoch {epoch:02d}/{args.epochs}: "
                  f"val accuracy={metrics['accuracy']:.4f}, "
                  f"worst group={metrics['worst_group_accuracy']:.4f}", flush=True)
            if epoch % 5 == 0 or epoch == args.epochs:
                torch.save(
                    {"epoch": epoch, "model": model.state_dict()},
                    args.output_dir / f"checkpoint_{epoch:03d}.pt",
                )
            if epoch % 5 == 0:
                visualizer.render(epoch)

    visualizer.save_timeline()
    # Plot the three main evidence streams with a shared epoch axis.
    with (args.output_dir / "epoch_metrics.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    for name in ("accuracy", "worst_group_accuracy", "aligned_accuracy", "conflicting_accuracy"):
        axes[0].plot(epochs, [float(row[name]) for row in rows], label=name)
    axes[0].legend()
    axes[0].set_ylabel("Classification accuracy")
    for name in ("probe_label_group_mean_accuracy", "probe_place_group_mean_accuracy"):
        axes[1].plot(epochs, [float(row[name]) for row in rows], label=name)
    axes[1].legend()
    axes[1].set_ylabel("Probe group-mean accuracy")
    axes[2].plot(epochs, [float(row["accuracy_gap"]) for row in rows])
    axes[2].set_ylabel("Aligned - conflicting")
    axes[2].set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(args.output_dir / "epoch_curves.png", dpi=150)
    plt.close(figure)

    test_metrics = evaluator.evaluate(model, data.test_loader())
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
