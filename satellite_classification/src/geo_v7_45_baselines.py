"""Train-only required comparisons; explicitly outside final-training timer."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms import v2

import geo_worker as base
import geo_v7_45_worker as worker


def baseline_model(name, classes):
    if name == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 512), nn.ReLU(),
                             nn.Dropout(.3), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, classes))
    if name != "cnn":
        raise ValueError("Baseline must be mlp or cnn")
    layers = []
    previous = 3
    for width in (16, 32, 64, 128):
        layers.extend([nn.Conv2d(previous, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
                       nn.ReLU(), nn.Conv2d(width, width, 3, padding=1, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)])
        previous = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.3), nn.Linear(128, classes))


def transform(training):
    ops = [v2.ToImage(), v2.Resize((64, 64), antialias=True)]
    if training:
        ops += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
                v2.RandomChoice([v2.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)])]
    return v2.Compose(ops + [v2.ToDtype(torch.float32, scale=True), v2.Normalize(base.MEAN, base.STD)])


def run_baselines(data_dir, output_dir, cache_dir, epochs=8, workers=2, allow_cpu=False):
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Positive baseline epochs required")
    device = base._device(dict(allow_cpu=allow_cpu))
    output = Path(output_dir) / "baselines"
    output.mkdir(parents=True, exist_ok=True)
    tables = worker.load_metadata(data_dir, include_test=False)
    rows = []
    for name in ("mlp", "cnn"):
        started = time.time()
        base.seed_everything(2026)  # Before initialization, including isolated reruns.
        model = baseline_model(name, len(tables["classes"])).to(device)
        cfg = dict(seed=2026, batch_size=128)
        job = dict(workers=workers)
        train = base._loader(worker.ImageDataset(tables, data_dir, "train", transform(True), cache_dir=cache_dir),
                             cfg, job, training=True)
        val = base._loader(worker.ImageDataset(tables, data_dir, "val", transform(False), cache_dir=cache_dir), cfg, job)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            count, correct, loss_total = 0, 0, 0.
            for images, labels in train:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite baseline loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 2.)
                scaler.step(optimizer)
                scaler.update()
                count += len(labels)
                correct += int(logits.detach().argmax(1).eq(labels).sum())
                loss_total += float(loss.detach()) * len(labels)
            if count != len(tables["train"]):
                raise RuntimeError("Baseline epoch must cover every training row")
            scheduler.step()
            model.eval()
            vcount, vcorrect, vloss = 0, 0, 0.
            with torch.inference_mode():
                for images, labels in val:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits = model(images)
                    vcount += len(labels)
                    vcorrect += int(logits.argmax(1).eq(labels).sum())
                    vloss += float(nn.functional.cross_entropy(logits.float(), labels, reduction="sum"))
            history.append(dict(epoch=epoch, train_loss=loss_total/count, train_accuracy=correct/count,
                                val_loss=vloss/vcount, val_accuracy=vcorrect/vcount))
            base._json_write(output / f"{name}_history.json", history)
            print(f"Baseline {name} {epoch}/{epochs}: val={vcorrect}/{vcount}", flush=True)
        weights = output / f"{name}_fp32.pt"
        torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        classes=tables["classes"], seed=2026, size=64, epochs=epochs,
                        data_signature=tables["data_signature"], trained_on="train"), weights)
        row = dict(model=name, seed=2026, epochs=epochs, train_images=len(tables["train"]),
                   validation_count=vcount, validation_correct=vcorrect, validation_accuracy=vcorrect/vcount,
                   validation_loss=vloss/vcount, elapsed_minutes=(time.time()-started)/60,
                   trained_on="train", weight_sha256=base._sha256(weights),
                   outside_final_training_timer=True, fit_test_images=0)
        rows.append(row)
        base._json_write(output / "results.json", rows)
        del model, optimizer, scheduler, scaler, train, val
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows
