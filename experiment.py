"""Train three scratch VGG11 Waterbirds models and track each epoch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import vgg11

from waterbirds_data import CORRELATIONS, correlation_name, prepare_synthetic_data


class WaterbirdsImages(Dataset):
    """Read generated file rows or Hugging Face Waterbirds rows."""

    def __init__(self, rows, root: Path | None = None, training: bool = False):
        self.rows = rows
        self.root = root
        self.display_transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
        transforms = [self.display_transform]
        if training:
            transforms.append(T.RandomHorizontalFlip())
        transforms.extend([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
        self.transform = T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, index: int) -> Image.Image:
        value = self.rows[index]["image"]
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if self.root is None:
            raise ValueError("A root directory is required for generated image paths")
        with Image.open(self.root / value) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int):
        row = self.rows[index]
        return self.transform(self._image(index)), int(row["label"]), int(row["place"])

    def display_image(self, index: int) -> Image.Image:
        return self.display_transform(self._image(index))


class VGG11BirdClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = vgg11(weights=None)
        self.network.classifier[6] = nn.Linear(4096, 2)
        self.network.features[19].inplace = False

    def forward_with_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.network.features(images)
        features = self.network.avgpool(features).flatten(1)
        hidden = self.network.classifier[:5](features)
        return self.network.classifier[5:](hidden), hidden

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(images)[0]

    @property
    def last_convolution(self) -> nn.Module:
        return self.network.features[18]


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, epochs: int):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        learning_rate = self.optimizer.param_groups[0]["lr"]
        total_loss = 0.0
        sample_count = 0
        for images, labels, _ in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(labels)
            sample_count += len(labels)
        self.scheduler.step()
        return {"train_loss": total_loss / sample_count, "learning_rate": learning_rate}


class EpochEvaluator:
    def __init__(self, device: torch.device):
        self.device = device

    def evaluate(self, model: VGG11BirdClassifier, loader: DataLoader,
                 features: bool = False):
        model.eval()
        predictions, labels, places, hidden = [], [], [], []
        with torch.inference_mode():
            for images, batch_labels, batch_places in loader:
                if features:
                    logits, batch_hidden = model.forward_with_features(images.to(self.device))
                    hidden.append(batch_hidden.cpu().numpy())
                else:
                    logits = model(images.to(self.device))
                predictions.append(logits.argmax(dim=1).cpu().numpy())
                labels.append(batch_labels.numpy())
                places.append(batch_places.numpy())
        if not labels:
            raise ValueError("Cannot evaluate an empty dataset")
        pred, y, p = map(np.concatenate, (predictions, labels, places))
        result = {"accuracy": float(np.mean(pred == y))}
        group_accuracies = []
        for label in (0, 1):
            for place in (0, 1):
                selected = (y == label) & (p == place)
                key = f"group_{label}{place}"
                result[f"{key}_count"] = int(selected.sum())
                result[f"{key}_accuracy"] = float(np.mean(pred[selected] == y[selected])) if selected.any() else float("nan")
                if selected.any():
                    group_accuracies.append(result[f"{key}_accuracy"])
        result["worst_group_accuracy"] = min(group_accuracies)
        if features:
            return result, (np.concatenate(hidden), y, p)
        return result


class LinearProbeEvaluator:
    def __init__(self, device: torch.device, seed: int):
        self.device = device
        self.seed = seed

    def extract(self, model: VGG11BirdClassifier, loader: DataLoader):
        model.eval()
        xs, ys, ps = [], [], []
        with torch.inference_mode():
            for images, labels, places in loader:
                _, hidden = model.forward_with_features(images.to(self.device))
                xs.append(hidden.cpu().numpy())
                ys.append(labels.numpy())
                ps.append(places.numpy())
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)

    def evaluate(self, train_data, test_data) -> dict[str, float]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        train_x, train_y, train_p = train_data
        test_x, test_y, test_p = test_data
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        train_groups = 2 * train_y + train_p
        test_groups = 2 * test_y + test_p
        counts = np.bincount(train_groups, minlength=4)
        if np.any(counts == 0) or np.any(np.bincount(test_groups, minlength=4) == 0):
            raise ValueError("All four groups are required for probes")
        weights = len(train_groups) / (4 * counts[train_groups])
        result = {}
        for name, target_train, target_test in (
            ("bird_probe_accuracy", train_y, test_y),
            ("background_probe_accuracy", train_p, test_p),
        ):
            probe = LogisticRegression(C=1.0, max_iter=1000, random_state=self.seed)
            probe.fit(train_x, target_train, sample_weight=weights)
            prediction = probe.predict(test_x)
            result[name] = float(np.mean([
                np.mean(prediction[test_groups == group] == target_test[test_groups == group])
                for group in range(4)
            ]))
        return result


class GradCAMVisualizer:
    """Save fixed true-class CAM thumbnails for every epoch."""

    def __init__(self, model: VGG11BirdClassifier, dataset: WaterbirdsImages,
                 sample_indices: dict[str, int], device: torch.device, output_dir: Path):
        self.model = model
        self.dataset = dataset
        self.sample_indices = sample_indices
        self.device = device
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, epoch: int) -> None:
        self.model.eval()
        activations = []

        def capture(_module, _inputs, output):
            activations.append(output)

        hook = self.model.last_convolution.register_forward_hook(capture)
        try:
            for sample, index in self.sample_indices.items():
                activations.clear()
                image, label, _ = self.dataset[index]
                with torch.enable_grad():
                    logits = self.model(image.unsqueeze(0).to(self.device))
                    gradient = torch.autograd.grad(logits[0, label], activations[0])[0]
                    weights = gradient.mean(dim=(2, 3), keepdim=True)
                    heatmap = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
                    heatmap = torch.nn.functional.interpolate(
                        heatmap, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
                heatmap = heatmap.detach().cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-8)
                original = np.asarray(self.dataset.display_image(index), dtype=np.float32) / 255
                color = matplotlib.colormaps["jet"](heatmap)[..., :3]
                overlay = Image.fromarray(((0.55 * original + 0.45 * color) * 255).astype(np.uint8))
                overlay.resize((128, 128), Image.Resampling.LANCZOS).save(
                    self.output_dir / f"epoch_{epoch:03d}_{sample}.png")
        finally:
            hook.remove()


def build_gradcam_grid(output_dir: Path, epochs: int) -> None:
    """Rows are epochs; columns are the three fixed samples for each model."""
    names = [correlation_name(c) for c in CORRELATIONS]
    cell, left, top = 128, 70, 38
    figure = Image.new("RGB", (left + 9 * cell, top + epochs * cell), "white")
    draw = ImageDraw.Draw(figure)
    font = ImageFont.load_default()
    columns = [(name, sample) for name in names for sample in "abc"]
    for column, (name, sample) in enumerate(columns):
        draw.text((left + column * cell + 4, 9), f"{name} / {sample}", fill="black", font=font)
    for epoch in range(1, epochs + 1):
        draw.text((7, top + (epoch - 1) * cell + 50), str(epoch), fill="black", font=font)
        for column, (name, sample) in enumerate(columns):
            path = output_dir / name / "cam" / f"epoch_{epoch:03d}_{sample}.png"
            with Image.open(path) as thumbnail:
                figure.paste(thumbnail, (left + column * cell, top + (epoch - 1) * cell))
    draw.text((5, 9), "Epoch", fill="black", font=font)
    figure.save(output_dir / "gradcam_grid.png")


def plot_curves(output_dir: Path) -> None:
    metrics = (
        ("validation_accuracy", "Validation accuracy"),
        ("test_accuracy", "Test accuracy"),
        ("bird_probe_accuracy", "Bird linear probe (group mean)"),
        ("background_probe_accuracy", "Background linear probe (group mean)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for corr in CORRELATIONS:
        name = correlation_name(corr)
        with (output_dir / name / "epoch_metrics.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        x = [int(row["epoch"]) for row in rows]
        for axis, (key, title) in zip(axes.flat, metrics):
            axis.plot(x, [float(row[key]) for row in rows], label=name)
            axis.set_title(title)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "epoch_curves.png", dpi=150)
    plt.close(figure)


def write_combined_metrics(output_dir: Path) -> None:
    rows = []
    for corr in CORRELATIONS:
        path = output_dir / correlation_name(corr) / "epoch_metrics.csv"
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    with (output_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _save_state(path: Path, epoch: int, model: nn.Module, trainer: Trainer) -> None:
    state = {"epoch": epoch, "model": model.state_dict(),
             "optimizer": trainer.optimizer.state_dict(), "scheduler": trainer.scheduler.state_dict(),
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _load_state(path: Path, model: nn.Module, trainer: Trainer) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["epoch"])


def run(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch-size must be positive; num-workers cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise ValueError(f"Output directory is not empty: {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    settings = {"epochs": args.epochs, "batch_size": args.batch_size,
                "num_workers": args.num_workers, "seed": args.seed,
                "device": str(device), "model": "vgg11", "weights": None,
                "test_dataset": "grodino/waterbirds:test"}
    settings_path = output_dir / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("Run settings do not match the existing output")
    else:
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    manifests = prepare_synthetic_data(args.cub_root, args.places_root, output_dir, args.seed)

    from datasets import load_dataset
    test = WaterbirdsImages(load_dataset("grodino/waterbirds", split="test"))
    groups = defaultdict(list)
    for index, (label, place) in enumerate(zip(test.rows["label"], test.rows["place"])):
        groups[(int(label), int(place))].append(index)
    rng = np.random.default_rng(args.seed)
    sample_indices = {"a": int(rng.choice(groups[(1, 1)])),
                      "b": int(rng.choice(groups[(1, 0)])),
                      "c": int(rng.choice(groups[(0, 1)]))}
    sample_path = output_dir / "gradcam_samples.json"
    if sample_path.exists() and json.loads(sample_path.read_text(encoding="utf-8")) != sample_indices:
        raise ValueError("Fixed GradCAM samples changed on resume")
    sample_path.write_text(json.dumps(sample_indices, indent=2), encoding="utf-8")
    for sample, index in sample_indices.items():
        test.display_image(index).save(output_dir / f"sample_{sample}.png")

    def loader(dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")

    test_loader = loader(test)
    evaluator = EpochEvaluator(device)
    probe = LinearProbeEvaluator(device, args.seed)
    for corr in CORRELATIONS:
        name = correlation_name(corr)
        condition_dir = output_dir / name
        condition_dir.mkdir(exist_ok=True)
        train = WaterbirdsImages(manifests[corr]["train"], output_dir, training=True)
        train_probe = WaterbirdsImages(manifests[corr]["train"], output_dir)
        validation = WaterbirdsImages(manifests[corr]["validation"], output_dir)
        train_loader = loader(train, shuffle=True)
        probe_loader = loader(train_probe)
        validation_loader = loader(validation)
        _seed_everything(args.seed)
        model = VGG11BirdClassifier().to(device)
        trainer = Trainer(model, device, args.epochs)
        visualizer = GradCAMVisualizer(model, test, sample_indices, device, condition_dir / "cam")
        latest_path = condition_dir / "latest.pt"
        start_epoch = _load_state(latest_path, model, trainer) if args.resume and latest_path.exists() else 0
        metrics_path = condition_dir / "epoch_metrics.csv"
        if metrics_path.exists():
            with metrics_path.open(newline="", encoding="utf-8") as file:
                old_rows = list(csv.DictReader(file))
            if len(old_rows) < start_epoch:
                raise ValueError(f"Metric and checkpoint epochs disagree for {name}")
            if start_epoch == 0:
                metrics_path.write_text("", encoding="utf-8")
            elif len(old_rows) > start_epoch:
                # A crash after the CSV write but before checkpoint commit leaves
                # one uncommitted row. Recompute it from the prior checkpoint.
                with metrics_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(old_rows[0]))
                    writer.writeheader()
                    writer.writerows(old_rows[:start_epoch])
        elif start_epoch:
            raise ValueError(f"Missing metrics for resumed {name}")
        with metrics_path.open("a", newline="", encoding="utf-8") as file:
            writer = None
            for epoch in range(start_epoch + 1, args.epochs + 1):
                row = {"epoch": epoch, "correlation": corr}
                row.update(trainer.train_epoch(train_loader))
                val_metrics = evaluator.evaluate(model, validation_loader)
                test_metrics, test_features = evaluator.evaluate(model, test_loader, features=True)
                train_features = probe.extract(model, probe_loader)
                row["validation_accuracy"] = val_metrics["accuracy"]
                row["test_accuracy"] = test_metrics["accuracy"]
                row["validation_worst_group_accuracy"] = val_metrics["worst_group_accuracy"]
                row["test_worst_group_accuracy"] = test_metrics["worst_group_accuracy"]
                row.update(probe.evaluate(train_features, test_features))
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    if start_epoch == 0:
                        writer.writeheader()
                writer.writerow(row)
                file.flush()
                visualizer.render(epoch)
                _save_state(latest_path, epoch, model, trainer)
                print(f"{name} epoch {epoch:02d}/{args.epochs}: val={row['validation_accuracy']:.4f} "
                      f"test={row['test_accuracy']:.4f}", flush=True)
        torch.save({"epoch": args.epochs, "correlation": corr, "model": model.state_dict()},
                   condition_dir / "final.pt")
    plot_curves(output_dir)
    write_combined_metrics(output_dir)
    build_gradcam_grid(output_dir, args.epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    raw = Path("../shortcut-learning-dynamics/data/raw")
    parser.add_argument("--cub-root", type=Path, default=raw / "CUB_200_2011")
    parser.add_argument("--places-root", type=Path, default=raw / "places365")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vgg11_correlations"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
