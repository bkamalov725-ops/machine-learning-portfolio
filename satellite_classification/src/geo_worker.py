"""Isolated, reproducible PyTorch trials. Run ``python geo_worker.py --job job.json``.

Screening uses only a deterministic split of train.csv. Its four-view score is
a synthetic robustness proxy, never an estimate measured on official val.csv.
Incomplete runs are deliberately non-resumable; use a new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models, transforms as T
from torchvision.transforms import functional as TF


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INNER_SPLIT_SEED = 20261003
MODEL_NAMES = {"mlp", "cnn_plain", "cnn_bn", "cnn_gap", "resnet18", "resnet34", "resnet50"}


def seed_everything(seed):
    """Seed this process; the scheduler must isolate its visible CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # May already be configured in an importing process.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _worker_seed(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_csv(path, columns):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != list(columns):
        raise ValueError(f"{Path(path).name}: expected columns {list(columns)}, got {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{Path(path).name}: table is empty")
    if "id" in frame:
        ids = frame["id"]
        invalid = ids.map(lambda x: not x or x in {".", ".."} or any(c in x for c in "/\\\x00:") or Path(x).name != x)
        if invalid.any() or ids.duplicated().any():
            raise ValueError(f"{Path(path).name}: IDs must be unique, literal filenames without paths")
    if "label" in frame:
        if not frame["label"].map(lambda x: bool(re.fullmatch(r"0|[1-9][0-9]*", x))).all():
            raise ValueError(f"{Path(path).name}: labels must be nonnegative integers")
        frame["label"] = frame["label"].astype(np.int64)
    return frame


def load_tables(data_dir, phase="final"):
    """Read validated metadata; ``phase='screen'`` never opens val/test metadata."""
    root = Path(data_dir)
    classes_frame = _read_csv(root / "classes.csv", ["label", "class_name"])
    classes_frame = classes_frame.sort_values("label").reset_index(drop=True)
    if classes_frame["label"].tolist() != list(range(len(classes_frame))):
        raise ValueError("classes.csv: class labels must be unique and contiguous from zero")
    if classes_frame["class_name"].duplicated().any() or (classes_frame["class_name"].str.strip() == "").any():
        raise ValueError("classes.csv: class names must be unique and nonempty")
    result = {"classes": classes_frame["class_name"].tolist()}
    names = ["train"] if phase == "screen" else ["train", "val", "test"]
    signatures = {"classes.csv": _sha256(root / "classes.csv")}
    split_ids, overlaps = {}, {}
    for split in names:
        columns = ["id"] if split == "test" else ["id", "label"]
        frame = _read_csv(root / f"{split}.csv", columns)
        if "label" in frame and not frame["label"].between(0, len(classes_frame) - 1).all():
            raise ValueError(f"{split}.csv: label outside classes.csv range")
        ids = set(frame["id"])
        # Filenames are split-local. Repeated names do not establish image leakage.
        for earlier, earlier_ids in split_ids.items():
            overlaps[f"{earlier}:{split}"] = len(earlier_ids.intersection(ids))
        split_ids[split] = ids
        result[split] = frame
        signatures[f"{split}.csv"] = _sha256(root / f"{split}.csv")
    sample_path = root / "sample_submission_example.csv"
    if phase != "screen" and sample_path.exists():
        sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        if list(sample.columns) != ["id", "label"]:
            raise ValueError("Sample submission must have columns id,label")
        result["sample_submission"] = sample
        result["sample_order_matches_test"] = sample["id"].tolist() == result["test"]["id"].tolist()
        signatures[sample_path.name] = _sha256(sample_path)
    result["data_signature"] = signatures
    result["split_id_overlaps"] = overlaps
    return result


def _size(cfg):
    return 32 if cfg["model"] == "mlp" else int(cfg.get("size", 96))


def build_transform(cfg, training=False, corruption=None):
    """Only torchvision transforms; evaluation corruptions have fixed parameters."""
    size = _size(cfg)
    steps = [T.Resize((size, size), antialias=True)]
    if training:
        augment = cfg.get("augment", "standard")
        if augment not in {"none", "basic", "standard", "robust"}:
            raise ValueError(f"Unknown augmentation: {augment}")
        if augment != "none":
            steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomChoice([T.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)]),
                      T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02)]
        if augment == "robust":
            steps += [T.RandomApply([T.GaussianBlur(5, sigma=(0.2, 1.5))], p=0.25),
                      T.RandomInvert(p=0.10),
                      T.RandomApply([T.Resize((size // 2, size // 2), antialias=True),
                                     T.Resize((size, size), antialias=True)], p=0.25)]
    elif corruption == "blur":
        steps += [T.GaussianBlur(5, sigma=(1.2, 1.2))]
    elif corruption == "lowres":
        steps += [T.Resize((size // 2, size // 2), antialias=True), T.Resize((size, size), antialias=True)]
    elif corruption == "invert":
        steps += [T.RandomInvert(p=1.0)]
    elif corruption not in {None, "clean", "none"}:
        raise ValueError(f"Unknown evaluation corruption: {corruption}")
    return T.Compose(steps + [T.ToTensor(), T.Normalize(MEAN, STD)])


class _ScreenViews:
    def __init__(self, cfg):
        self.transforms = [build_transform(cfg, corruption=view) for view in ("clean", "blur", "lowres", "invert")]

    def __call__(self, image):
        return torch.stack([transform(image) for transform in self.transforms])


class _PredictionViews:
    def __init__(self, cfg, tta):
        if tta not in {"none", "d4", "fivecrop"}:
            raise ValueError("tta must be none, d4, or fivecrop")
        self.tta, self.size = tta, _size(cfg)
        self.normal = build_transform(cfg)
        self.tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __call__(self, image):
        if self.tta == "fivecrop":
            expanded = TF.resize(image, [math.ceil(self.size * 1.14)] * 2, antialias=True)
            return torch.stack([self.tensor(crop) for crop in TF.five_crop(expanded, self.size)])
        tensor = self.normal(image)
        if self.tta == "none":
            return tensor.unsqueeze(0)
        return torch.stack([TF.rotate(view, angle) for view in (tensor, TF.hflip(tensor)) for angle in (0, 90, 180, 270)])


class GeoDataset(Dataset):
    """Lazy image decoding; IDs are preserved verbatim and never guessed."""
    def __init__(self, frame, data_dir, split, transform):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported image split: {split}")
        self.frame = frame.reset_index(drop=True).copy()
        self.root = (Path(data_dir) / split).resolve()
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = (self.root / str(row["id"])).resolve()
        if path.parent != self.root:
            raise ValueError("Image ID resolves outside its split directory")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]) if "label" in self.frame else -1


def build_model(cfg, num_classes, pretrained=True):
    """Build a fresh model; prediction uses pretrained=False and cannot download."""
    name = cfg["model"]
    dropout = float(cfg.get("dropout", 0.3))
    if name not in MODEL_NAMES or num_classes < 2:
        raise ValueError(f"Unsupported model or class count: {name}, {num_classes}")
    if name == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 512), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))
    if name.startswith("cnn_"):
        layers, channels = [], 3
        widths = (32, 64, 128, 256) if name == "cnn_gap" else (32, 64, 128)
        for width in widths:
            layers.append(nn.Conv2d(channels, width, 3, padding=1, bias=name == "cnn_plain"))
            if name != "cnn_plain":
                layers.append(nn.BatchNorm2d(width))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        grid = 1 if name == "cnn_gap" else 2
        layers += [nn.AdaptiveAvgPool2d(grid), nn.Flatten()]
        if name == "cnn_gap":
            layers += [nn.Dropout(dropout), nn.Linear(channels, num_classes)]
        else:
            layers += [nn.Linear(channels * grid * grid, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)]
        return nn.Sequential(*layers)
    weights_enum = getattr(models, name.replace("resnet", "ResNet") + "_Weights")
    weights_name = "IMAGENET1K_V2" if name == "resnet50" else "IMAGENET1K_V1"
    weights = getattr(weights_enum, weights_name) if pretrained else None
    backbone = getattr(models, name)(weights=weights)
    features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    scope = cfg.get("unfreeze", "layer4")
    if scope not in {"head", "layer4", "layer3", "all"}:
        raise ValueError("unfreeze must be head, layer4, layer3, or all")
    for parameter_name, parameter in backbone.named_parameters():
        parameter.requires_grad = scope == "all" or (scope in {"layer3", "layer4"} and parameter_name.startswith("layer4.")) or (scope == "layer3" and parameter_name.startswith("layer3."))
    return nn.Sequential(backbone, nn.Sequential(nn.Linear(features, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_classes)))


def _device(job):
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if job.get("allow_cpu", False):
        return torch.device("cpu")
    raise RuntimeError("CUDA is required. CPU is permitted only when allow_cpu=true is explicitly set.")


def _loader(dataset, cfg, job, training=False):
    workers = max(0, min(2, int(job.get("workers", 2))))
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    return DataLoader(dataset, batch_size=int(cfg.get("batch_size", 64)), shuffle=training,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0, worker_init_fn=_worker_seed, generator=generator)


def _inner_split(frame):
    rng = np.random.default_rng(INNER_SPLIT_SEED)
    train_indices, valid_indices = [], []
    for _label, rows in frame.groupby("label", sort=True):
        indices = rng.permutation(rows.index.to_numpy())
        if len(indices) < 2:
            raise ValueError("Every class needs at least two training rows for stratified screening")
        count = max(1, min(len(indices) - 1, int(round(len(indices) * 0.15))))
        valid_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    return frame.loc[sorted(train_indices)].reset_index(drop=True), frame.loc[sorted(valid_indices)].reset_index(drop=True)


def _optimizer(model, cfg):
    learning_rate = float(cfg.get("lr", 0.001))
    if cfg["model"].startswith("resnet"):
        backbone = [p for p in model[0].parameters() if p.requires_grad]
        groups = ([{"params": backbone, "lr": learning_rate * 0.1}] if backbone else [])
        groups.append({"params": model[1].parameters(), "lr": learning_rate})
    else:
        groups = [{"params": model.parameters(), "lr": learning_rate}]
    common = {"lr": learning_rate, "weight_decay": float(cfg.get("weight_decay", 0.0001))}
    name = cfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, **common)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True, **common)
    raise ValueError(f"Unsupported optimizer: {name}")


def _set_train(model, original_trainability, head_only):
    for parameter, enabled in original_trainability:
        parameter.requires_grad_(enabled and not head_only)
    model.train()
    # Frozen BatchNorm must not silently update ImageNet running statistics.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


@torch.inference_mode()
def _evaluate(model, loader, device, view_names):
    model.eval()
    counts, losses, total = np.zeros(len(view_names)), np.zeros(len(view_names)), 0
    for images, labels in loader:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        labels = labels.to(device, non_blocking=True)
        total += len(labels)
        for view in range(images.shape[1]):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images[:, view].to(device, non_blocking=True))
                loss = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation output")
            counts[view] += (logits.argmax(1) == labels).sum().item()
            losses[view] += loss.item()
    return {name: {"accuracy": float(counts[i] / total), "loss": float(losses[i] / total)} for i, name in enumerate(view_names)}


def _versions():
    return {"torch": str(torch.__version__), "torchvision": torchvision.__version__, "numpy": np.__version__, "pandas": pd.__version__}


def train(job):
    """Fit an independent trial, select its best epoch, and atomically save outputs."""
    started = time.monotonic()
    cfg, phase = dict(job["config"]), job.get("phase", "screen")
    if phase not in {"screen", "final"}:
        raise ValueError("Training phase must be screen or final")
    epochs = int(cfg.get("epochs", 10))
    if epochs < 1 or int(cfg.get("batch_size", 64)) < 1 or _size(cfg) < 16:
        raise ValueError("epochs/batch_size must be positive and image size at least 16")
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = _device(job)
    tables = load_tables(job["data_dir"], phase=phase)
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("config") == cfg and previous.get("phase") == phase and previous.get("seed") == seed and previous.get("data_signature") == tables["data_signature"] and (output / "best.pt").exists():
            return previous
        raise RuntimeError("Existing result does not exactly match config, phase, seed, and CSV fingerprints")
    if any((output / name).exists() for name in ("best.pt", "best.pt.tmp", "history.json", "history.json.tmp")):
        raise RuntimeError("Incomplete run is non-resumable; choose a new output directory")
    if phase == "screen":
        training_frame, valid_frame = _inner_split(tables["train"])
        valid_dataset = GeoDataset(valid_frame, job["data_dir"], "train", _ScreenViews(cfg))
        view_names = ("clean", "blur", "lowres", "invert")
    else:
        training_frame = tables["train"]
        valid_dataset = GeoDataset(tables["val"], job["data_dir"], "val", build_transform(cfg))
        view_names = ("clean",)
    training = _loader(GeoDataset(training_frame, job["data_dir"], "train", build_transform(cfg, training=True)), cfg, job, True)
    validation = _loader(valid_dataset, cfg, job)
    model = build_model(cfg, len(tables["classes"]), pretrained=True).to(device)
    optimizer = _optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    backbone_flags = [(p, p.requires_grad) for p in model[0].parameters()] if cfg["model"].startswith("resnet") else []
    warmup = max(0, int(cfg.get("warmup_epochs", 1))) if backbone_flags else 0
    history, best_metric, best_loss, best_epoch, best_views = [], -math.inf, math.inf, 0, None
    for epoch in range(1, epochs + 1):
        _set_train(model, backbone_flags, epoch <= warmup)
        total, correct, loss_sum = 0, 0, 0.0
        for images, labels in training:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += len(labels)
            correct += (logits.detach().argmax(1) == labels).sum().item()
            loss_sum += loss.item() * len(labels)
        views = _evaluate(model, validation, device, view_names)
        metric = float(np.mean([value["accuracy"] for value in views.values()]))
        eval_loss = float(np.mean([value["loss"] for value in views.values()]))
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "selection_metric": metric, "evaluation": views, "learning_rates": [group["lr"] for group in optimizer.param_groups]}
        history.append(row)
        # Screening ties preserve the earlier epoch: selection uses the proxy alone.
        improved = metric > best_metric or (phase == "final" and metric == best_metric and eval_loss < best_loss)
        if improved:
            best_metric, best_loss, best_epoch, best_views = metric, eval_loss, epoch, views
            checkpoint = {"state_dict": model.state_dict(), "config": cfg, "num_classes": len(tables["classes"]),
                          "classes": tables["classes"], "epoch": epoch, "selection_metric": metric,
                          "phase": phase, "data_signature": tables["data_signature"]}
            torch.save(checkpoint, output / "best.pt.tmp")
            os.replace(output / "best.pt.tmp", output / "best.pt")
        _json_write(output / "history.json", history)
        scheduler.step()
        print(json.dumps({"trial_id": cfg["trial_id"], "phase": phase, "epoch": epoch, "selection_metric": metric}), flush=True)
    result = {"trial_id": cfg["trial_id"], "model": cfg["model"], "seed": seed, "phase": phase,
              "selection_metric": best_metric, "best_epoch": best_epoch, "elapsed_seconds": time.monotonic() - started,
              "config": cfg, "versions": _versions(), "data_signature": tables["data_signature"],
              "checkpoint": str((output / "best.pt").resolve()), "best_evaluation": best_views,
              "selection_metric_name": "synthetic_inner_four_view_accuracy" if phase == "screen" else "official_val_accuracy"}
    result["inner_accuracy" if phase == "screen" else "val_accuracy"] = best_views["clean"]["accuracy"]
    if phase == "screen":
        result["inner_balanced_accuracy"] = best_metric
        result["inner_split_seed"] = INNER_SPLIT_SEED
        result["metric_note"] = "Synthetic corruption proxy on a held-out training subset; no official validation data used."
    _json_write(result_path, result)
    return result


@torch.inference_mode()
def predict_job(job):
    """Write ordered probabilities, averaging softmax probabilities across TTA views."""
    checkpoint_path = Path(job["checkpoint"])
    if job.get("checkpoint_sha256") and _sha256(checkpoint_path) != job["checkpoint_sha256"]:
        raise ValueError("Checkpoint contents changed after prediction job was scheduled")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("phase") != "final":
        raise ValueError("Only final-stage checkpoints may predict official val/test sets")
    cfg = checkpoint["config"]
    seed_everything(int(cfg["seed"]))
    device = _device(job)
    split, tta = job.get("split", "test"), job.get("tta", "none")
    if split not in {"val", "test"}:
        raise ValueError("Prediction split must be val or test")
    tables = load_tables(job["data_dir"], phase="final")
    if checkpoint["classes"] != tables["classes"] or checkpoint["num_classes"] != len(tables["classes"]):
        raise ValueError("Checkpoint class mapping differs from current classes.csv")
    for name in ("classes.csv", "train.csv", f"{split}.csv"):
        if checkpoint.get("data_signature", {}).get(name) != tables["data_signature"][name]:
            raise ValueError(f"Checkpoint CSV fingerprint differs: {name}")
    model = build_model(cfg, checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    frame = tables[split]
    loader = _loader(GeoDataset(frame, job["data_dir"], split, _PredictionViews(cfg, tta)), cfg, job)
    batches = []
    for images, _labels in loader:
        probabilities = None
        for view in range(images.shape[1]):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images[:, view].to(device, non_blocking=True))
            if logits.ndim != 2 or logits.shape[1] != checkpoint["num_classes"] or not torch.isfinite(logits).all():
                raise ValueError("Invalid prediction logits or class count")
            current = logits.float().softmax(dim=1).cpu()
            probabilities = current if probabilities is None else probabilities + current
        batches.append((probabilities / images.shape[1]).numpy())
    probabilities = np.concatenate(batches).astype(np.float32)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Predictions must be finite normalized probabilities")
    prefix = Path(job.get("prefix", str(Path(job["output_dir"]) / f"{split}_{tta}")))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"ids": frame["id"].to_numpy(dtype=str), "probabilities": probabilities,
              "classes": np.asarray(checkpoint["classes"], dtype=str),
              "class_ids": np.arange(checkpoint["num_classes"], dtype=np.int64)}
    metadata = {"tta": tta, "split": split, "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": _sha256(checkpoint_path), "num_classes": checkpoint["num_classes"],
                "classes": checkpoint["classes"], "class_ids": list(range(checkpoint["num_classes"])),
                "count": len(frame), "data_signature": tables["data_signature"], "versions": _versions()}
    if split == "val":
        arrays["labels"] = frame["label"].to_numpy(dtype=np.int64)
        metadata["accuracy"] = float((probabilities.argmax(1) == arrays["labels"]).mean())
    npz_path = Path(str(prefix) + ".npz")
    temporary = Path(str(npz_path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, npz_path)
    _json_write(Path(str(prefix) + ".json"), metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8-sig"))
    result = predict_job(job) if job.get("phase") == "predict" else train(job)
    print(json.dumps({"status": "ok", "phase": job.get("phase", "screen"),
                      "output_dir": job["output_dir"], "selection_metric": result.get("selection_metric"),
                      "accuracy": result.get("accuracy")}), flush=True)


if __name__ == "__main__":
    main()
