"""Bounded final training, separate prediction, and FP32 last-full-epoch EMA.

Only train and explicitly selected labeled validation rows are fitted. Test is
read solely by prediction/cache-for-prediction jobs, never by a training job.
Screening fits train only. Final fitting uses supplied labeled train + val.
Test metadata/images are inaccessible from a training job. Screening is not
part of the separately measured final-fitting window.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
from torchvision import models
from torchvision.transforms import v2
from torchvision.transforms import functional as TF

import geo_worker as base


PROTOCOL = "v7_45_trainval_separate_predict"
MODELS = ("resnet152", "resnet101", "resnet50", "inception_v3", "googlenet", "vgg16_bn")
SEEDS = (20261003, 20261104)
MEAN, STD = base.MEAN, base.STD


class BudgetExpired(RuntimeError):
    pass


def check_deadline(job):
    deadline = float(job.get("deadline_unix", job.get("deadline_epoch", math.inf)))
    if time.time() >= deadline:
        raise BudgetExpired("Worker training deadline expired")


def config_for_model(model, seed, trial_id=None, crop_scale=(.55, 1.0)):
    cfg = dict(model=model, seed=int(seed), trial_id=trial_id or f"v7_45_{model}_seed{seed}",
               epochs=14, minimum_epochs=3, size=299 if model == "inception_v3" else 224,
               batch_size=16 if model == "resnet152" else 24, val_repeat=2,
               lr_backbone=.0002, lr_head=.002, weight_decay=.05, dropout=.3,
               label_smoothing=.1, mix_p=.3, mixup_alpha=.4, cutmix_alpha=1.0,
               ema_decay=.999, grad_clip=2.0, warmup_epochs=1,
               crop_scale=list(crop_scale), freeze_early=model.startswith("resnet"),
               inception_transform_input=model == "inception_v3",
               augment="torchvision_v2_corruptions", optimizer="adamw")
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg.get("model") not in MODELS:
        raise ValueError("Only the six explicitly supported torchvision CNN backbones are permitted")
    for name in ("epochs", "minimum_epochs", "size", "batch_size", "val_repeat", "warmup_epochs"):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["size"] < 75 or (cfg["model"] == "inception_v3" and cfg["size"] < 299):
        raise ValueError("Inputs must support the selected torchvision backbone")
    if cfg.get("optimizer") != "adamw" or cfg.get("augment") != "torchvision_v2_corruptions":
        raise ValueError("Fixed AdamW/official torchvision augmentation protocol required")
    if cfg["minimum_epochs"] > cfg["epochs"]:
        raise ValueError("minimum_epochs must not exceed epochs")
    for name in ("lr_backbone", "lr_head", "weight_decay", "dropout", "label_smoothing",
                 "mix_p", "ema_decay", "grad_clip", "mixup_alpha", "cutmix_alpha"):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if cfg["lr_backbone"] <= 0 or cfg["lr_head"] <= 0 or cfg["grad_clip"] <= 0:
        raise ValueError("Positive learning rates and gradient limit are required")
    if cfg["weight_decay"] < 0 or cfg["mixup_alpha"] <= 0 or cfg["cutmix_alpha"] <= 0:
        raise ValueError("Invalid decay or mixing alpha")
    for name in ("dropout", "label_smoothing", "mix_p", "ema_decay"):
        if not 0 <= cfg[name] < 1:
            raise ValueError(f"{name} must lie in [0,1)")
    scale = cfg.get("crop_scale")
    if not isinstance(scale, (tuple, list)) or len(scale) != 2 or not 0 < scale[0] <= scale[1] <= 1:
        raise ValueError("crop_scale must be a pair within (0,1]")
    if not isinstance(cfg.get("freeze_early"), bool):
        raise ValueError("freeze_early must be explicit")
    if cfg["model"] == "inception_v3" and cfg["freeze_early"]:
        raise ValueError("This independent Inception configuration does not freeze early layers")
    if cfg.get("inception_transform_input") is not (cfg["model"] == "inception_v3"):
        raise ValueError("Inception must retain its official pretrained internal input convention")


def load_metadata(data_dir, include_test=False):
    """Training calls never open test.csv, test images or test cache arrays."""
    root = Path(data_dir)
    classes = base._read_csv(root / "classes.csv", ["label", "class_name"]).sort_values("label")
    if classes["label"].tolist() != list(range(len(classes))) or len(classes) < 2:
        raise ValueError("Class labels must be contiguous and contain at least two classes")
    if classes["class_name"].duplicated().any() or (classes["class_name"].str.strip() == "").any():
        raise ValueError("Class names must be unique and nonempty")
    result = {"classes": classes["class_name"].tolist(), "data_signature": {}}
    names = ["train", "val"] + (["test"] if include_test else [])
    for split in names:
        path = root / f"{split}.csv"
        frame = base._read_csv(path, ["id"] if split == "test" else ["id", "label"])
        if "label" in frame and not frame["label"].between(0, len(classes) - 1).all():
            raise ValueError(f"{split}: label outside class range")
        result[split] = frame
        result["data_signature"][path.name] = base._sha256(path)
    result["data_signature"]["classes.csv"] = base._sha256(root / "classes.csv")
    return result


def _indices(values, count, name):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be an explicit sequence")
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError(f"{name} must contain integer indices")
    values = [int(x) for x in values]
    if len(set(values)) != len(values) or any(x < 0 or x >= count for x in values):
        raise ValueError(f"{name}: duplicate or out-of-range validation index")
    return values


def validate_partition(job, count):
    fitted = _indices(job.get("train_val_indices", list(range(count)) if job.get("phase") == "final" else []), count, "train_val_indices")
    heldout = _indices(job.get("holdout_val_indices", list(range(count)) if job.get("phase") == "screen" else []), count, "holdout_val_indices")
    if set(fitted).intersection(heldout):
        raise ValueError("Held-out validation rows must never be fitted")
    if set(fitted).union(heldout) != set(range(count)):
        raise ValueError("Fitted and held-out indices must partition all validation rows")
    if job.get("phase") == "screen":
        if fitted or len(heldout) != count:
            raise ValueError("Screening fits train only; all val rows must stay held out")
    elif job.get("phase") == "final":
        if heldout or len(fitted) != count:
            raise ValueError("Final fitting uses every labeled validation row and no holdout")
    else:
        raise ValueError("Training phase must be screen or final")
    return fitted, heldout


def job_signature(job):
    content = {key: job.get(key) for key in ("config", "phase", "min_epochs", "train_val_indices", "holdout_val_indices",
              "split", "indices", "tta", "checkpoint_sha256", "data_signature")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build_cache(data_dir, cache_dir, side=320, include_test=False, deadline_unix=math.inf):
    """Deterministic per-image resizing, not dataset-fitted preprocessing.

    The controller must call include_test=True only after the recipe is frozen.
    Cache arrays contain private images and MUST NOT enter recovery archives.
    """
    if not isinstance(side, int) or side < 299:
        raise ValueError("Cache side must be at least 299")
    tables = load_metadata(data_dir, include_test=include_test)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "protocol": PROTOCOL, "side": side, "splits": {}, "data_signature": {}}
    if manifest.get("protocol") != PROTOCOL or manifest.get("side") != side:
        raise ValueError("Existing image cache protocol/size differs")
    for key, value in tables["data_signature"].items():
        if key in manifest["data_signature"] and manifest["data_signature"][key] != value:
            raise ValueError("Existing image cache dataset differs")
    splits = ["train", "val"] + (["test"] if include_test else [])
    for split in splits:
        check_deadline({"deadline_unix": deadline_unix})
        target = root / f"{split}.npy"
        shape = (len(tables[split]), 3, side, side)
        if target.exists() and manifest["splits"].get(split, {}).get("shape") == list(shape):
            existing = np.load(target, mmap_mode="r", allow_pickle=False)
            if existing.shape != shape or existing.dtype != np.uint8:
                raise ValueError("Existing cache array shape/dtype mismatch")
            del existing
            continue
        temporary = target.with_name(target.name + ".tmp")
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
        try:
            for index, image_id in enumerate(tables[split]["id"]):
                check_deadline({"deadline_unix": deadline_unix})
                with Image.open(Path(data_dir) / split / image_id) as image:
                    resized = TF.resize(image.convert("RGB"), [side, side], antialias=True)
                    array[index] = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
            array.flush()
        finally:
            del array
        os.replace(temporary, target)
        manifest["splits"][split] = {"shape": list(shape), "dtype": "uint8",
            "table_sha256": tables["data_signature"][f"{split}.csv"]}
        manifest["data_signature"].update(tables["data_signature"])
        base._json_write(manifest_path, manifest)
    return manifest


def build_transform(cfg, training=False):
    size = int(cfg["size"])
    if not training:
        return v2.Compose([v2.ToImage(), v2.Resize((size, size), antialias=True),
                           v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
    if not hasattr(v2, "JPEG") or not hasattr(v2, "GaussianNoise"):
        raise RuntimeError("torchvision v2 JPEG/GaussianNoise are required (official 0.22 or later)")
    return v2.Compose([
        v2.ToImage(), v2.RandomResizedCrop((size, size), scale=tuple(cfg["crop_scale"]),
                                         ratio=(.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
        v2.RandomChoice([v2.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)]),
        v2.ColorJitter(.2, .2, .2, .02),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(.2, 1.5))], p=.25),
        v2.RandomApply([v2.JPEG(quality=(8, 45))], p=.30),
        v2.RandomApply([v2.Resize((size // 2, size // 2), antialias=True),
                        v2.Resize((size, size), antialias=True)], p=.25),
        v2.RandomGrayscale(p=.10), v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply([v2.GaussianNoise(mean=0., sigma=.03, clip=True)], p=.15),
        v2.Normalize(MEAN, STD)])


class ImageDataset(Dataset):
    """Split-local rows, optional read-only memory map; never infer filename labels."""
    def __init__(self, tables, data_dir, split, transform, indices=None, cache_dir=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        self.frame = tables[split]
        self.indices = list(range(len(self.frame))) if indices is None else list(indices)
        self.root = (Path(data_dir) / split).resolve()
        self.split, self.transform = split, transform
        self.cache_path, self.cache = None, None
        if cache_dir:
            cache_root = Path(cache_dir)
            manifest = json.loads((cache_root / "cache_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("protocol") != PROTOCOL:
                raise ValueError("Image cache protocol differs")
            for key in ("classes.csv", f"{split}.csv"):
                if manifest.get("data_signature", {}).get(key) != tables["data_signature"][key]:
                    raise ValueError("Image cache metadata differs")
            self.cache_path = cache_root / f"{split}.npy"
            array = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
            expected = tuple(manifest["splits"][split]["shape"])
            if array.shape != expected or array.dtype != np.uint8 or array.shape[0] != len(self.frame):
                raise ValueError("Image cache array shape/dtype differs")
            del array

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        row = self.frame.iloc[original]
        if self.cache_path:
            if self.cache is None:
                self.cache = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
            # A private contiguous writable copy is necessary for torchvision tensor ops.
            image = torch.from_numpy(np.array(self.cache[original], copy=True))
        else:
            path = (self.root / str(row["id"])).resolve()
            if path.parent != self.root:
                raise ValueError("Image ID resolves outside split directory")
            with Image.open(path) as source:
                image = source.convert("RGB")
                image = self.transform(image)
            return image, int(row["label"]) if "label" in self.frame else -1
        return self.transform(image), int(row["label"]) if "label" in self.frame else -1


def build_model(cfg, num_classes, pretrained=True):
    validate_config(cfg)
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("At least two classes required")
    name = cfg["model"]
    if name.startswith("resnet"):
        enum = {"resnet152": models.ResNet152_Weights, "resnet101": models.ResNet101_Weights,
                "resnet50": models.ResNet50_Weights}[name]
        network = getattr(models, name)(weights=enum.IMAGENET1K_V2 if pretrained else None)
        width = network.fc.in_features
        network.fc = nn.Identity()
        if cfg["freeze_early"]:
            for block in (network.conv1, network.bn1, network.layer1, network.layer2):
                block.requires_grad_(False)
    elif name == "inception_v3":
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        network = models.inception_v3(weights=weights, aux_logits=bool(pretrained),
                                      transform_input=cfg["inception_transform_input"], init_weights=False)
        network.aux_logits, network.AuxLogits = False, None
        width = network.fc.in_features
        network.fc, network.dropout = nn.Identity(), nn.Identity()
    elif name == "googlenet":
        weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
        network = models.googlenet(weights=weights, aux_logits=bool(pretrained),
                                  transform_input=True, init_weights=False)
        network.aux_logits, network.aux1, network.aux2 = False, None, None
        width = network.fc.in_features
        network.fc, network.dropout = nn.Identity(), nn.Identity()
    elif name == "vgg16_bn":
        original = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        # The 1000-class ImageNet classifier is not reused. The supplied CNN
        # feature extractor is followed by our own compact sequential head.
        network = nn.Sequential(original.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        width = 512
        if cfg["freeze_early"]:
            for block in original.features[:24]:
                block.requires_grad_(False)
    return nn.Sequential(network, nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(width, num_classes)))


def set_train(model):
    model.train()
    # Parameters and running mean/variance are frozen together in early ResNet blocks.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                parameter.requires_grad for parameter in module.parameters()):
            module.eval()


def optimizer_for(model, cfg):
    groups = []
    for part, rate in ((model[0], cfg["lr_backbone"]), (model[1], cfg["lr_head"])):
        for decay in (False, True):
            params = [parameter for parameter in part.parameters() if parameter.requires_grad
                      and (parameter.ndim > 1) == decay]
            if params:
                groups.append(dict(params=params, lr=rate, initial_lr=rate,
                                   weight_decay=cfg["weight_decay"] if decay else 0.0))
    return torch.optim.AdamW(groups)


class WarmEMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = float(decay), 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()
        for key, averaged in self.model.state_dict().items():
            if averaged.is_floating_point():
                averaged.lerp_(current[key], 1. - decay)
            else:
                averaged.copy_(current[key])


def _save_checkpoint(path, model, cfg, tables, job, epoch, fit_indices, holdout_indices, updates):
    state = {key: value.detach().cpu().float().clone() if value.is_floating_point()
             else value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint = dict(state_dict=state, config=cfg, classes=tables["classes"],
                      num_classes=len(tables["classes"]), data_signature=tables["data_signature"],
                      worker_protocol=PROTOCOL, job_signature=job_signature(job), phase=job["phase"],
                      epoch=int(epoch), weight_kind="fixed_last_ema", ema_updates=updates,
                      train_val_indices=fit_indices, holdout_val_indices=holdout_indices,
                      source_signature=source_signature(),
                      minimum_epochs=int(job.get("min_epochs", cfg["minimum_epochs"])),
                      complete_protocol=(epoch >= int(job.get("min_epochs", cfg["minimum_epochs"]))
                          if job["phase"] == "final" else epoch == cfg["epochs"]))
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def source_signature():
    return {"geo_v7_45_worker.py": base._sha256(Path(__file__)),
            "geo_worker.py": base._sha256(Path(base.__file__))}


def _training_result(job, cfg, tables, checkpoint, epochs, fitted, heldout, skipped, steps,
                     started, stopped_reason=None, training_finished=False):
    eligible = (epochs >= int(job.get("min_epochs", cfg["minimum_epochs"]))
                if job["phase"] == "final" else epochs == cfg["epochs"])
    exists = checkpoint.exists()
    return dict(worker_protocol=PROTOCOL, complete_protocol=eligible, completed=eligible,
        complete=eligible, eligible=eligible, training_finished=training_finished,
        epochs_done=epochs, completed_epochs=epochs, epochs_requested=cfg["epochs"],
        minimum_epochs=int(job.get("min_epochs", cfg["minimum_epochs"])),
        stopped_early=bool(stopped_reason), stopped_reason=stopped_reason,
        phase=job["phase"], config=cfg, job_signature=job_signature(job),
        source_signature=source_signature(), data_signature=tables["data_signature"],
        classes=tables["classes"], checkpoint=str(checkpoint) if exists else None,
        checkpoint_sha256=base._sha256(checkpoint) if exists else None,
        weight_kind="fixed_last_ema", last_epoch=epochs, train_images=len(tables["train"]),
        fit_val_images=len(fitted), holdout_val_images=len(heldout),
        training_rows_per_epoch=len(tables["train"]) + len(fitted) * cfg["val_repeat"],
        train_val_indices=fitted, holdout_val_indices=heldout, heldout_evaluation=None,
        training_performed=True, fit_test_images=0, skipped_steps=skipped, successful_steps=steps,
        elapsed_seconds=time.time() - started)


@torch.inference_mode()
def _evaluate(model, loader, device, job):
    model.eval()
    total, correct, loss = 0, 0, 0.
    for images, labels in loader:
        check_deadline(job)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite held-out logits")
        loss += float(nn.functional.cross_entropy(logits.float(), labels, reduction="sum"))
        correct += int(logits.argmax(1).eq(labels).sum())
        total += len(labels)
    return dict(count=total, correct=correct, accuracy=correct / total, loss=loss / total) if total else None


def train(job):
    started, cfg = time.time(), dict(job["config"])
    validate_config(cfg)
    minimum = int(job.get("min_epochs", cfg["minimum_epochs"]))
    if not 1 <= minimum <= cfg["epochs"]:
        raise ValueError("Declared minimum must lie between one and the maximum epoch count")
    check_deadline(job)
    base.seed_everything(cfg["seed"])
    device = base._device(job)
    tables = load_metadata(job["data_dir"], include_test=False)
    fit_indices, holdout_indices = validate_partition(job, len(tables["val"]))
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best.pt", "checkpoint_a.pt", "checkpoint_b.pt", "history.json", "result.json")):
        raise FileExistsError("Fits must start in a fresh attempt; cached/partial weights are never silently reused")
    model = build_model(cfg, len(tables["classes"]), pretrained=True).to(device)
    check_deadline(job)
    optimizer = optimizer_for(model, cfg)
    ema = WarmEMA(model, cfg["ema_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    transform = build_transform(cfg, training=True)
    cache = job.get("cache_dir")
    pieces = [ImageDataset(tables, job["data_dir"], "train", transform, cache_dir=cache)]
    if fit_indices:
        pieces.extend(ImageDataset(tables, job["data_dir"], "val", transform, fit_indices, cache)
                      for _ in range(cfg["val_repeat"]))
    dataset = ConcatDataset(pieces)
    loader = base._loader(dataset, cfg, job, training=True)
    if loader.drop_last:
        raise ValueError("Every labeled training row must be visited; drop_last forbidden")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    mixup = v2.MixUp(alpha=cfg["mixup_alpha"], num_classes=len(tables["classes"]))
    cutmix = v2.CutMix(alpha=cfg["cutmix_alpha"], num_classes=len(tables["classes"]))
    total_planned_steps = cfg["epochs"] * len(loader)
    warmup_steps = max(1, cfg["warmup_epochs"] * len(loader))
    history, skipped, steps, stopped = [], 0, 0, False
    checkpoint = output / "checkpoint_a.pt"
    expected = len(tables["train"]) + len(fit_indices) * cfg["val_repeat"]
    for epoch in range(1, cfg["epochs"] + 1):
        seen, loss_sum, correct, mixed = 0, 0., 0, 0
        try:
            check_deadline(job)
            set_train(model)
            for images, labels in loader:
                check_deadline(job)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                reference_labels = labels
                if len(labels) > 1 and torch.rand((), device=device) < cfg["mix_p"]:
                    images, labels = (mixup if torch.rand((), device=device) < .5 else cutmix)(images, labels)
                    mixed += 1
                factor = ((steps + 1) / warmup_steps if steps < warmup_steps else
                          .5 * (1. + math.cos(math.pi * min(1., (steps - warmup_steps) /
                                  max(1, total_planned_steps - warmup_steps)))))
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * factor
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite training output")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                if device.type != "cuda" and not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite CPU gradient")
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                successful = scaler.get_scale() >= old_scale
                if successful:
                    ema.update(model)
                    steps += 1
                else:
                    skipped += 1
                seen += len(reference_labels)
                loss_sum += float(loss.detach()) * len(reference_labels)
                correct += int(logits.detach().argmax(1).eq(reference_labels).sum())
                check_deadline(job)
            if seen != expected:
                raise RuntimeError("Completed epoch did not visit every planned labeled row")
        except BudgetExpired:
            stopped = True
            break
        # Save only the last COMPLETED epoch. Interrupted batches must not enter
        # either checkpoint or the accompanying result. No validation prediction
        # is performed in this training function (screen/final both train only).
        # The preceding result always points to the OTHER slot. An external
        # deadline stop during saving therefore cannot corrupt its committed
        # checkpoint/result pair. At most two full FP32 snapshots are retained.
        checkpoint = output / ("checkpoint_a.pt" if epoch % 2 else "checkpoint_b.pt")
        _save_checkpoint(checkpoint, ema.model, cfg, tables, job, epoch,
                         fit_indices, holdout_indices, ema.updates)
        row = dict(epoch=epoch, train_loss=loss_sum / seen,
                   train_accuracy_against_pre_mix_labels=correct / seen, training_rows=seen,
                   unique_train_images=len(tables["train"]), unique_fit_val_images=len(fit_indices),
                   val_repeat=cfg["val_repeat"], mixed_batches=mixed, skipped_steps_total=skipped,
                   successful_steps_total=steps, ema_updates=ema.updates,
                   seconds=time.time() - started,
                   learning_rates=[group["lr"] for group in optimizer.param_groups])
        history.append(row)
        base._json_write(output / "history.json", history)
        snapshot = _training_result(job, cfg, tables, checkpoint, len(history),
                                    fit_indices, holdout_indices, skipped, steps, started)
        base._json_write(output / "result.json", snapshot)
        print(json.dumps(dict(epoch=epoch, loss=row["train_loss"], seconds=row["seconds"])), flush=True)
    result = _training_result(job, cfg, tables, checkpoint, len(history), fit_indices,
                              holdout_indices, skipped, steps, started,
                              stopped_reason="deadline" if stopped else None,
                              training_finished=True)
    base._json_write(output / "result.json", result)
    return result


def d4_views(images, tta):
    if tta == "none":
        return [images]
    if tta != "d4":
        raise ValueError("Inference supports none or complete eight-view D4")
    return [TF.rotate(value, angle) for value in (images, TF.hflip(images))
            for angle in (0, 90, 180, 270)]


def _load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def predict(job):
    started, cfg = time.time(), dict(job["config"])
    validate_config(cfg)
    check_deadline(job)
    if job.get("split") not in {"val", "test"}:
        raise ValueError("Prediction supports held-out val or test only")
    base.seed_everything(cfg["seed"])
    device = base._device(job)
    tables = load_metadata(job["data_dir"], include_test=job["split"] == "test")
    digest = base._sha256(job["checkpoint"])
    if job.get("checkpoint_sha256") and job["checkpoint_sha256"] != digest:
        raise ValueError("Checkpoint hash differs")
    checkpoint = _load_checkpoint(job["checkpoint"])
    if checkpoint.get("worker_protocol") != PROTOCOL or not checkpoint.get("complete_protocol"):
        raise ValueError("Only eligible full-epoch checkpoints may predict")
    if checkpoint.get("source_signature") != source_signature():
        raise ValueError("Checkpoint worker source differs; rerun the declared training code")
    if checkpoint.get("config") != cfg or checkpoint.get("classes") != tables["classes"]:
        raise ValueError("Checkpoint configuration/class order differs")
    for key, value in checkpoint["data_signature"].items():
        if tables["data_signature"].get(key) != value:
            raise ValueError("Checkpoint labeled dataset signature differs")
    split = job["split"]
    indices = _indices(job.get("indices", list(range(len(tables[split])))), len(tables[split]), "indices")
    if not indices:
        raise ValueError("Prediction indices must not be empty")
    if split == "val":
        if checkpoint["phase"] != "screen" or set(indices) != set(checkpoint["holdout_val_indices"]):
            raise ValueError("Validation prediction must use the train-only screening model's unseen val set")
    model = build_model(cfg, len(tables["classes"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    dataset = ImageDataset(tables, job["data_dir"], split, build_transform(cfg),
                           indices, job.get("cache_dir"))
    loader = base._loader(dataset, cfg, job)
    probabilities = []
    with torch.inference_mode():
        for images, _labels in loader:
            check_deadline(job)
            images = images.to(device, non_blocking=True)
            sum_probs = None
            views = d4_views(images, job.get("tta", "none"))
            for view in views:
                check_deadline(job)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(view)
                value = logits.float().softmax(dim=1)
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Nonfinite prediction")
                sum_probs = value if sum_probs is None else sum_probs + value
            probabilities.append((sum_probs / len(views)).cpu().numpy())
    probs = np.concatenate(probabilities).astype(np.float32)
    if probs.shape != (len(indices), len(tables["classes"])) or not np.isfinite(probs).all():
        raise ValueError("Prediction shape or values differ")
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "probs.npz").exists() or (output / "result.json").exists():
        raise FileExistsError("Prediction attempt output already exists")
    path = output / "probs.npz"
    temporary = path.with_name(path.name + ".tmp")
    frame = tables[split].iloc[indices]
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, probs=probs, ids=np.asarray(frame["id"].tolist(), dtype=str),
                            labels=frame["label"].to_numpy(dtype=np.int64) if split == "val" else
                                   np.full(len(indices), -1, dtype=np.int64),
                            classes=np.asarray(tables["classes"], dtype=str), indices=np.asarray(indices),
                            class_labels=np.arange(len(tables["classes"]), dtype=np.int64),
                            data_signature=json.dumps(tables["data_signature"], sort_keys=True),
                            config_json=json.dumps(cfg, sort_keys=True), checkpoint_sha256=digest,
                            job_signature=job_signature(job), worker_protocol=PROTOCOL)
    os.replace(temporary, path)
    result = dict(worker_protocol=PROTOCOL, complete_protocol=True, completed=True, complete=True,
                  eligible=True, usable=True,
                  phase="predict", config=cfg, split=split, indices=indices, tta=job.get("tta", "none"),
                  rows=len(indices), forward_views=8 if job.get("tta") == "d4" else 1,
                  probabilities=str(path), probs_path=str(path), probs_sha256=base._sha256(path),
                  checkpoint_sha256=digest, data_signature=tables["data_signature"],
                  source_signature=source_signature(),
                  classes=tables["classes"], job_signature=job_signature(job),
                  elapsed_seconds=time.time() - started, training_performed=False)
    base._json_write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    if job.get("mode") == "predict" or job.get("phase") == "predict":
        result = predict(job)
    else:
        result = train(job)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

