"""Synthetic-only worker checks. No real competition tables/images or downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

import geo_v7_45_worker as worker


def tables(include_test=False):
    value = dict(classes=["a", "b"],
        train=pd.DataFrame({"id": ["t0.png", "t1.png", "t2.png"], "label": [0, 1, 0]}),
        val=pd.DataFrame({"id": ["v0.png", "v1.png"], "label": [0, 1]}),
        data_signature={"classes.csv": "cls", "train.csv": "tr", "val.csv": "va"})
    if include_test:
        value["test"] = pd.DataFrame({"id": ["x0.png", "x1.png"]})
        value["data_signature"]["test.csv"] = "te"
    return value


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, data_dir, split, transform, indices=None, cache_dir=None):
        self.frame = metadata[split]
        self.indices = list(range(len(self.frame))) if indices is None else list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        original = self.indices[index]
        row = self.frame.iloc[original]
        return torch.full((3, 8, 8), original / 3.), int(row.get("label", -1))


def tiny_model(cfg, classes, pretrained=True):
    return nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten()), nn.Linear(4, classes))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = worker.config_for_model("resnet152", 1)

    def small_cfg(self, epochs=2, minimum=1):
        return dict(self.cfg, epochs=epochs, minimum_epochs=minimum, batch_size=2,
                    val_repeat=1, mix_p=0.)

    def test_six_configs_with_fixed_declared_policy(self):
        for model in worker.MODELS:
            cfg = worker.config_for_model(model, 4)
            self.assertEqual(cfg["epochs"], 14)
            self.assertEqual(cfg["minimum_epochs"], 3)
            self.assertEqual(cfg["val_repeat"], 2)
            self.assertEqual(cfg["size"], 299 if model == "inception_v3" else 224)
        with self.assertRaises(ValueError):
            worker.config_for_model("vit_b_16", 1)
        with self.assertRaises(ValueError):
            worker.validate_config(dict(self.cfg, minimum_epochs=15))

    def test_partitions_screen_trainonly_final_all_labeled(self):
        self.assertEqual(worker.validate_partition(dict(phase="screen"), 2), ([], [0, 1]))
        self.assertEqual(worker.validate_partition(dict(phase="final"), 2), ([0, 1], []))
        for bad in (dict(phase="final", train_val_indices=[0], holdout_val_indices=[1]),
                    dict(phase="screen", train_val_indices=[0], holdout_val_indices=[1]),
                    dict(phase="final", train_val_indices=[0, 0], holdout_val_indices=[]),
                    dict(phase="unknown")):
            with self.assertRaises(ValueError):
                worker.validate_partition(bad, 2)

    def test_training_metadata_never_opens_test(self):
        opened = []
        def read(path, columns):
            opened.append(Path(path).name)
            if Path(path).name == "classes.csv":
                return pd.DataFrame({"label": [0, 1], "class_name": ["a", "b"]})
            return tables()[Path(path).stem]
        with mock.patch.object(worker.base, "_read_csv", side_effect=read), \
             mock.patch.object(worker.base, "_sha256", return_value="synthetic"):
            metadata = worker.load_metadata("unused")
        self.assertEqual(set(opened), {"classes.csv", "train.csv", "val.csv"})
        self.assertNotIn("test", metadata)

    def test_official_transforms_uint8_cpu_and_float_finite(self):
        cfg = dict(self.cfg, size=80)
        transform = worker.build_transform(cfg, training=True)
        torch.manual_seed(8)
        for _ in range(4):
            result = transform(torch.randint(0, 256, (3, 90, 90), dtype=torch.uint8))
            self.assertEqual(tuple(result.shape), (3, 80, 80))
            self.assertEqual(result.dtype, torch.float32)
            self.assertTrue(torch.isfinite(result).all())

    def test_resnet50_correct_enum_and_frozen_bn(self):
        class TinyResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 1)
                self.bn1 = nn.BatchNorm2d(4)
                self.layer1 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
                self.layer2 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
                self.layer3 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
                self.fc = nn.Linear(4, 1000)
            def forward(self, image):
                x = self.layer3(self.layer2(self.layer1(self.bn1(self.conv1(image)))))
                return self.fc(x.mean((2, 3)))
        cfg = worker.config_for_model("resnet50", 1)
        with mock.patch.object(worker.models, "resnet50", return_value=TinyResNet()) as factory:
            model = worker.build_model(cfg, 2, pretrained=True)
        self.assertEqual(factory.call_args.kwargs["weights"], worker.models.ResNet50_Weights.IMAGENET1K_V2)
        worker.set_train(model)
        self.assertFalse(model[0].bn1.training)
        self.assertFalse(model[0].layer2[1].training)
        self.assertTrue(model[0].layer3[1].training)
        before = model[0].bn1.running_mean.clone()
        model(torch.randn(2, 3, 8, 8))
        self.assertTrue(torch.equal(before, model[0].bn1.running_mean))

    def test_inception_and_googlenet_use_correct_constructor(self):
        class TinyNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 1000)
                self.dropout = nn.Dropout(.5)
            def forward(self, image):
                return self.fc(self.dropout(image.mean((2, 3))))
        for name in ("inception_v3", "googlenet"):
            with mock.patch.object(worker.models, name, return_value=TinyNetwork()) as factory:
                model = worker.build_model(worker.config_for_model(name, 3), 2, pretrained=False)
            self.assertTrue(factory.call_args.kwargs["transform_input"])
            self.assertFalse(model[0].aux_logits)
            self.assertIsInstance(model, nn.Sequential)
            self.assertEqual(tuple(model(torch.randn(2, 3, 8, 8)).shape), (2, 2))

    def test_vgg_has_own_compact_head_not_original_classifier(self):
        original = nn.Module()
        original.features = nn.Sequential(nn.Conv2d(3, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        original.classifier = nn.Linear(100, 1000)
        cfg = worker.config_for_model("vgg16_bn", 3)
        with mock.patch.object(worker.models, "vgg16_bn", return_value=original):
            model = worker.build_model(cfg, 2, pretrained=False)
        self.assertEqual(model[1][1].in_features, 512)
        self.assertEqual(tuple(model(torch.randn(2, 3, 8, 8)).shape), (2, 2))
        self.assertNotIn(original.classifier, list(model.modules()))

    def test_optimizer_no_decay_1d_and_ema_warmup(self):
        model = tiny_model(self.cfg, 2)
        optimizer = worker.optimizer_for(model, self.cfg)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertEqual(group["weight_decay"], .05 if parameter.ndim > 1 else 0.)
        ema = worker.WarmEMA(model, .999)
        before = ema.model[1].weight.clone()
        with torch.no_grad():
            model[1].weight.add_(1.)
        ema.update(model)
        self.assertEqual(ema.updates, 1)
        self.assertTrue(torch.allclose(ema.model[1].weight, before + 1 - 2/11))

    def test_d4_eight_distinct_views(self):
        image = torch.arange(12.).reshape(1, 3, 2, 2)
        views = worker.d4_views(image, "d4")
        self.assertEqual(len({item.numpy().tobytes() for item in views}), 8)
        with self.assertRaises(ValueError):
            worker.d4_views(image, "fit_balance")

    def test_screen_full_training_and_separate_validation_prediction(self):
        cfg = self.small_cfg()
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(worker, "load_metadata", side_effect=lambda root, include_test=False: tables(include_test)), \
             mock.patch.object(worker, "ImageDataset", TinyDataset), \
             mock.patch.object(worker, "build_model", side_effect=tiny_model), \
             mock.patch.object(worker, "_evaluate", side_effect=AssertionError("Training must not predict")):
            output = Path(temporary)
            job = dict(config=cfg, phase="screen", data_dir="unused", output_dir=str(output / "train"),
                       allow_cpu=True, workers=0)
            result = worker.train(job)
            self.assertTrue(result["completed"])
            self.assertEqual(result["completed_epochs"], 2)
            self.assertEqual(result["training_rows_per_epoch"], 3)
            self.assertEqual(result["fit_val_images"], 0)
            self.assertEqual(result["fit_test_images"], 0)
            self.assertIsNone(result["heldout_evaluation"])
            checkpoint = worker._load_checkpoint(result["checkpoint"])
            self.assertTrue(all(not x.is_floating_point() or x.dtype == torch.float32
                                for x in checkpoint["state_dict"].values()))
            prediction = worker.predict(dict(config=cfg, split="val", tta="d4", data_dir="unused",
                checkpoint=result["checkpoint"], checkpoint_sha256=result["checkpoint_sha256"],
                output_dir=str(output / "predict"), allow_cpu=True, workers=0))
            with np.load(prediction["probs_path"], allow_pickle=False) as saved:
                self.assertEqual(saved["ids"].tolist(), ["v0.png", "v1.png"])
                self.assertEqual(saved["classes"].tolist(), ["a", "b"])
                self.assertEqual(saved["class_labels"].tolist(), [0, 1])
                self.assertTrue(np.allclose(saved["probs"].sum(1), 1.))
            with self.assertRaises(FileExistsError):
                worker.train(job)

    def test_final_deadline_after_three_full_epochs_discards_partial_fourth(self):
        cfg = self.small_cfg(epochs=5, minimum=3)
        calls = 0
        def deadline(_job):
            nonlocal calls
            calls += 1
            if calls >= 27:
                raise worker.BudgetExpired("synthetic deadline")
        snapshots = []
        original_json_write = worker.base._json_write
        def capture(path, value):
            if Path(path).name == "result.json":
                snapshots.append(dict(value))
            original_json_write(path, value)
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(worker, "load_metadata", return_value=tables()), \
             mock.patch.object(worker, "ImageDataset", TinyDataset), \
             mock.patch.object(worker, "build_model", side_effect=tiny_model), \
             mock.patch.object(worker, "check_deadline", side_effect=deadline), \
             mock.patch.object(worker.base, "_json_write", side_effect=capture):
            result = worker.train(dict(config=cfg, phase="final", data_dir="unused",
                output_dir=temporary, allow_cpu=True, workers=0))
            self.assertEqual(result["epochs_done"], 3)
            self.assertTrue(result["eligible"])
            self.assertEqual(result["stopped_reason"], "deadline")
            checkpoint = worker._load_checkpoint(result["checkpoint"])
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertTrue(checkpoint["complete_protocol"])
            self.assertEqual(checkpoint["ema_updates"], 9)
            self.assertTrue(any(snapshot["eligible"] and not snapshot["training_finished"]
                                for snapshot in snapshots))

    def test_deadline_less_than_minimum_is_ineligible(self):
        cfg = self.small_cfg(epochs=5, minimum=3)
        calls = 0
        def deadline(_job):
            nonlocal calls
            calls += 1
            if calls >= 10:
                raise worker.BudgetExpired("synthetic deadline")
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(worker, "load_metadata", return_value=tables()), \
             mock.patch.object(worker, "ImageDataset", TinyDataset), \
             mock.patch.object(worker, "build_model", side_effect=tiny_model), \
             mock.patch.object(worker, "check_deadline", side_effect=deadline):
            result = worker.train(dict(config=cfg, phase="final", data_dir="unused",
                output_dir=temporary, allow_cpu=True, workers=0))
            self.assertEqual(result["epochs_done"], 1)
            self.assertFalse(result["eligible"])
            self.assertFalse(worker._load_checkpoint(result["checkpoint"])["complete_protocol"])

    def test_final_predict_cannot_report_validation(self):
        cfg = self.small_cfg(epochs=1)
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(worker, "load_metadata", side_effect=lambda root, include_test=False: tables(include_test)), \
             mock.patch.object(worker, "ImageDataset", TinyDataset), \
             mock.patch.object(worker, "build_model", side_effect=tiny_model):
            root = Path(temporary)
            trained = worker.train(dict(config=cfg, phase="final", data_dir="unused",
                output_dir=str(root / "train"), allow_cpu=True, workers=0))
            job = dict(config=cfg, data_dir="unused", checkpoint=trained["checkpoint"],
                       output_dir=str(root / "val"), split="val", allow_cpu=True, workers=0)
            with self.assertRaises(ValueError):
                worker.predict(job)
            result = worker.predict(dict(job, split="test", output_dir=str(root / "test")))
            self.assertFalse(result["training_performed"])
            with np.load(result["probs_path"], allow_pickle=False) as values:
                self.assertEqual(values["labels"].tolist(), [-1, -1])
            with self.assertRaises(ValueError):
                worker.predict(dict(job, split="test", checkpoint_sha256="bad"))
            with mock.patch.object(worker, "source_signature", return_value={}):
                with self.assertRaises(ValueError):
                    worker.predict(dict(job, split="test"))

    def test_stop_during_new_slot_save_keeps_previous_committed_pair(self):
        cfg = self.small_cfg(epochs=3)
        original_save = worker._save_checkpoint
        def interrupt_after_write(path, model, config, metadata, job, epoch, fitted, heldout, updates):
            original_save(path, model, config, metadata, job, epoch, fitted, heldout, updates)
            if epoch == 2:
                raise RuntimeError("Synthetic external interruption between checkpoint and result")
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(worker, "load_metadata", return_value=tables()), \
             mock.patch.object(worker, "ImageDataset", TinyDataset), \
             mock.patch.object(worker, "build_model", side_effect=tiny_model), \
             mock.patch.object(worker, "_save_checkpoint", side_effect=interrupt_after_write):
            with self.assertRaises(RuntimeError):
                worker.train(dict(config=cfg, phase="final", data_dir="unused",
                    output_dir=temporary, allow_cpu=True, workers=0))
            committed = json.loads((Path(temporary) / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(committed["completed_epochs"], 1)
            self.assertTrue(committed["eligible"])
            self.assertEqual(worker.base._sha256(committed["checkpoint"]), committed["checkpoint_sha256"])
            self.assertEqual(worker._load_checkpoint(committed["checkpoint"])["epoch"], 1)

    def test_amp_skipped_step_does_not_update_ema_or_schedule_count(self):
        class FakeScaler:
            def __init__(self):
                self.scale_value = 128.
                self.calls = 0
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                pass
            def get_scale(self):
                return self.scale_value
            def step(self, optimizer):
                if self.calls:
                    optimizer.step()
            def update(self):
                if not self.calls:
                    self.scale_value /= 2
                self.calls += 1
        cfg = self.small_cfg(epochs=1)
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(worker, "load_metadata", return_value=tables()), \
             mock.patch.object(worker, "ImageDataset", TinyDataset), \
             mock.patch.object(worker, "build_model", side_effect=tiny_model), \
             mock.patch.object(worker.torch.amp, "GradScaler", return_value=FakeScaler()):
            result = worker.train(dict(config=cfg, phase="screen", data_dir="unused",
                output_dir=temporary, allow_cpu=True, workers=0))
            self.assertEqual(result["successful_steps"], 1)
            self.assertEqual(result["skipped_steps"], 1)
            self.assertEqual(worker._load_checkpoint(result["checkpoint"])["ema_updates"], 1)

    def test_cache_synthetic_images_no_test_before_prediction(self):
        metadata = tables()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                (root / split).mkdir()
                for image_id in metadata[split]["id"]:
                    Image.fromarray(np.full((10, 12, 3), 100, np.uint8)).save(root / split / image_id)
            with mock.patch.object(worker, "load_metadata", return_value=metadata):
                manifest = worker.build_cache(root, root / "cache", side=299)
            self.assertEqual(set(manifest["splits"]), {"train", "val"})
            self.assertFalse((root / "cache" / "test.npy").exists())
            dataset = worker.ImageDataset(metadata, root, "train", worker.build_transform(self.cfg),
                cache_dir=root / "cache")
            self.assertEqual(tuple(dataset[0][0].shape), (3, 224, 224))
            # Windows cannot remove a file while its mmap is still referenced.
            del dataset


if __name__ == "__main__":
    unittest.main()
