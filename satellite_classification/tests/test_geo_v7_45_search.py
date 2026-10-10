"""Synthetic timing/protocol tests. Never accesses competition files."""
from pathlib import Path
from types import SimpleNamespace
import io
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import geo_v7_45_search as search


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def pool(self):
        p = SimpleNamespace(output_dir=self.run, data_dir=self.root / "data", code_dir=self.root / "code",
                            cache_dir=self.root / "cache", source_hashes={"a": "hash"}, data_signature={"val.csv": "valhash"},
                            environment={"fake": True}, started=0, classes=["a", "b"], failures=[], events=[])
        p.event = lambda *args, **kwargs: p.events.append((args, kwargs))
        p.val = pd.DataFrame({"id": ["v0", "v1"], "label": [0, 1]})
        return p

    def freeze(self, p, weights=None):
        weights = weights or {"r152": 1.}
        selected = dict(name="test_none", weights=weights, tta="none", correct=2, count=2, accuracy=1., loss=.1)
        recipe = dict(protocol=search.PROTOCOL, selected=selected, configs={key: {"epochs": 14} for key in weights},
                      source_hashes=p.source_hashes, data_signature=p.data_signature)
        recipe["recipe_fingerprint"] = search.fingerprint(recipe)
        search.write_json(p.output_dir / "selected_recipe.json", recipe)
        return recipe

    def final_run(self, p, recipe, *, missing=False):
        def run(jobs, deadline, reuse, attempt_root):
            self.assertFalse(reuse)
            self.assertEqual(deadline, 1000 + search.TRAIN_STOP_SECONDS)
            result = []
            for job in jobs:
                checkpoint = Path(attempt_root) / (job["key"] + ".pt")
                checkpoint.write_bytes(b"fake FP32 last epoch")
                result.append(dict(key=job["key"], config=job["config"], checkpoint=str(checkpoint),
                                   checkpoint_sha256=search.digest(checkpoint), completed_epochs=6, weight_kind="fixed_last_ema",
                                   result_dir=str(attempt_root)))
            return result[:-1] if missing else result
        p.run = run

    def test_recipes_limit_contributors(self):
        self.assertEqual(len(search.recipes()), 6)
        for _, weights in search.recipes():
            self.assertLessEqual(len(weights), 2)
            self.assertAlmostEqual(sum(weights.values()), 1.)

    def test_invalid_probability_rejected(self):
        for value in ([[np.nan, 1]], [[-.2, 1.2]], [[.2, .2]], [1, 2]):
            with self.assertRaises(ValueError):
                search.probabilities(value)

    def test_mixture_missing_or_three_keys_rejected(self):
        with self.assertRaises(ValueError):
            search.mixture({}, {"a": .3, "b": .3, "c": .4})
        with self.assertRaises(ValueError):
            search.mixture({"a": [[1, 0]]}, {"a": .9})

    def test_selection_checks_12_fixed_candidates(self):
        pred = np.array([[.8, .2], [.2, .8]])
        selected, table = search.choose_recipe({tta: {key: pred for key in ("r152", "r101", "inc")} for tta in search.TTAS}, [0, 1])
        self.assertEqual(len(table), 12)
        self.assertEqual(selected["correct"], 2)
        self.assertEqual(selected["tta"], "none")

    def test_final_timer_fresh_and_actual_epochs(self):
        p = self.pool()
        recipe = self.freeze(p)
        self.final_run(p, recipe)
        with patch.object(search.time, "time", return_value=1000), patch.object(search.time, "monotonic", side_effect=[0, 100]):
            fits = search.run_final_train(p)
        self.assertEqual(len(fits), 1)
        timer = search.read_json(self.run / "final_training_time.json")
        self.assertEqual(timer["elapsed_seconds"], 100)
        self.assertTrue(timer["within_limit"])
        self.assertFalse(timer["final_weight_reuse"])
        replay = search.read_json(self.run / "reproduce_final_recipe.json")
        self.assertEqual(replay["actual_completed_epochs"], {"r152": 6})
        search.verify_final(p)

    def test_failed_new_final_invalidates_old_csv_recoverably(self):
        p = self.pool()
        recipe = self.freeze(p)
        (self.run / search.SUBMISSION_NAME).write_text("old generated csv", encoding="utf-8")
        self.final_run(p, recipe, missing=True)
        with patch.object(search.time, "time", return_value=1000), patch.object(search.time, "monotonic", side_effect=[0, 30]):
            with self.assertRaisesRegex(RuntimeError, "Not all selected"):
                search.run_final_train(p)
        self.assertFalse((self.run / search.SUBMISSION_NAME).exists())
        old = list((self.run / "final_attempts").glob("*/previous_output/" + search.SUBMISSION_NAME))
        self.assertEqual(len(old), 1)
        self.assertEqual(old[0].read_text(), "old generated csv")

    def test_over_45_minutes_rejects_no_final_manifest(self):
        p = self.pool()
        recipe = self.freeze(p)
        self.final_run(p, recipe)
        with patch.object(search.time, "time", return_value=1000), patch.object(search.time, "monotonic", side_effect=[0, 2701]):
            with self.assertRaisesRegex(RuntimeError, "exceeded 45"):
                search.run_final_train(p)
        self.assertFalse((self.run / "frozen_final.json").exists())

    def test_current_timer_must_equal_frozen_attempt(self):
        p = self.pool()
        recipe = self.freeze(p)
        self.final_run(p, recipe)
        with patch.object(search.time, "time", return_value=1000), patch.object(search.time, "monotonic", side_effect=[0, 90]):
            search.run_final_train(p)
        timer = search.read_json(self.run / "final_training_time.json")
        timer["status"] = "failed"
        search.write_json(self.run / "final_training_time.json", timer)
        with self.assertRaisesRegex(RuntimeError, "older final"):
            search.verify_final(p)

    def test_predict_cannot_touch_test_before_final_guards(self):
        p = self.pool()
        with patch.object(search, "test_metadata") as test_meta, patch.object(search.worker, "build_cache") as cache:
            with self.assertRaises(FileNotFoundError):
                search.predict_final(p)
            test_meta.assert_not_called()
            cache.assert_not_called()

    def test_recipe_fingerprint_changed_rejected(self):
        p = self.pool()
        recipe = self.freeze(p)
        recipe["selected"]["weights"] = {"inc": 1.}
        search.write_json(self.run / "selected_recipe.json", recipe)
        with self.assertRaises(RuntimeError):
            search.verify_frozen_recipe(p)

    def test_sample_target_and_order_preserved(self):
        p = self.pool()
        p.data_dir.mkdir()
        pd.DataFrame({"id": ["b.jpg", "a.jpg"]}).to_csv(p.data_dir / "test.csv", index=False)
        pd.DataFrame({"id": ["a.jpg", "b.jpg"], "target": [0, 0]}).to_csv(p.data_dir / "sample_submission_example.csv", index=False)
        self.assertEqual(search.test_metadata(p), (["b.jpg", "a.jpg"], ["a.jpg", "b.jpg"], "target"))

    def test_checkpoint_hash_changed_rejected(self):
        p = self.pool()
        recipe = self.freeze(p)
        self.final_run(p, recipe)
        with patch.object(search.time, "time", return_value=1000), patch.object(search.time, "monotonic", side_effect=[0, 90]):
            fits = search.run_final_train(p)
        Path(fits[0]["checkpoint"]).write_bytes(b"modified")
        with self.assertRaisesRegex(RuntimeError, "checkpoint identity"):
            search.verify_final(p)

    def test_bundle_excludes_image_and_array_caches(self):
        p = self.pool()
        p.code_dir.mkdir()
        for name in search.CODE_FILES:
            (p.code_dir / name).write_text("# synthetic own code", encoding="utf-8")
        (self.run / "model.pt").write_bytes(b"synthetic checkpoint")
        (self.run / "image.jpg").write_bytes(b"must not archive")
        (self.run / "images.npy").write_bytes(b"must not archive")
        search.bundle(p)
        with tarfile.open(self.run / search.ARCHIVE_NAME) as archive:
            names = archive.getnames()
        self.assertIn("run/model.pt", names)
        self.assertNotIn("run/image.jpg", names)
        self.assertNotIn("run/images.npy", names)

    def test_crashed_worker_snapshot_not_accepted(self):
        p = object.__new__(search.Pool)
        p.output_dir, p.code_dir, p.data_dir, p.cache_dir = self.run, self.root, self.root, self.root
        p.workers, p.gpus, p.failures = 0, [0], []
        p.identity = lambda payload: payload
        p.event = lambda *a, **kw: None
        p.completed = lambda *a, **kw: {"key": "r152", "completed_epochs": 3}
        fake_process = SimpleNamespace(poll=lambda: 1, returncode=1)
        with patch.object(search.subprocess, "Popen", return_value=fake_process):
            result = p.run([dict(key="r152", phase="final", config={})], reuse=False)
        self.assertEqual(result, [])
        self.assertEqual(p.failures[0]["returncode"], 1)

    def test_completed_exit_zero_allowed(self):
        p = object.__new__(search.Pool)
        p.output_dir, p.code_dir, p.data_dir, p.cache_dir = self.run, self.root, self.root, self.root
        p.workers, p.gpus, p.failures = 0, [0], []
        p.identity = lambda payload: payload
        p.event = lambda *a, **kw: None
        p.completed = lambda *a, **kw: {"key": "r152", "completed_epochs": 14}
        fake_process = SimpleNamespace(poll=lambda: 0, returncode=0)
        with patch.object(search.subprocess, "Popen", return_value=fake_process):
            result = p.run([dict(key="r152", phase="final", config={})], reuse=False)
        self.assertEqual(len(result), 1)

    def test_bounded_stop_escalates_after_four_seconds(self):
        process = SimpleNamespace(pid=123, poll=lambda: None, terminate=lambda: None, kill=lambda: None)
        process.wait = unittest.mock.Mock(side_effect=[subprocess.TimeoutExpired("fake", 4), 0])
        with patch.object(search.os, "name", "nt"):
            search.stop_process(process)
        self.assertEqual(process.wait.call_args_list[0].kwargs["timeout"], 4)
        self.assertEqual(process.wait.call_args_list[1].kwargs["timeout"], 4)

    def complete_fixture(self):
        p = object.__new__(search.Pool)
        p.classes = ["a", "b"]
        p.train = pd.DataFrame({"id": ["t"]})
        p.val = pd.DataFrame({"id": ["v"]})
        p.source_hashes = {"geo_worker.py": "basehash", "geo_v7_45_worker.py": "workerhash"}
        p.data_signature = {"train.csv": "trainhash"}
        directory = self.run / "synthetic_fit"
        directory.mkdir()
        checkpoint = directory / "checkpoint_a.pt"
        checkpoint.write_bytes(b"fake archived FP32 weights")
        identity = dict(key="r152", phase="final", config={"epochs": 14})
        search.write_json(directory / "controller_identity.json", identity)
        search.write_json(directory / "job.json", identity)
        result = dict(config=identity["config"], complete_protocol=True, eligible=True, classes=p.classes,
                      worker_protocol=search.worker.PROTOCOL, job_signature=search.worker.job_signature(identity),
                      phase="final", data_signature=dict(p.data_signature), source_signature=dict(p.source_hashes),
                      completed_epochs=3, checkpoint=str(checkpoint), checkpoint_sha256=search.digest(checkpoint),
                      weight_kind="fixed_last_ema", train_images=1, fit_val_images=1, fit_test_images=0)
        search.write_json(directory / "result.json", result)
        return p, directory, identity, result

    def test_atomic_checkpoint_slot_accepted(self):
        p, directory, identity, _ = self.complete_fixture()
        record = p.completed(directory, identity)
        self.assertEqual(Path(record["checkpoint"]).name, "checkpoint_a.pt")

    def test_partial_epochs_or_source_change_rejected(self):
        p, directory, identity, result = self.complete_fixture()
        result["completed_epochs"] = 2
        search.write_json(directory / "result.json", result)
        self.assertIsNone(p.completed(directory, identity))
        result["completed_epochs"] = 3
        result["source_signature"]["geo_worker.py"] = "changed"
        search.write_json(directory / "result.json", result)
        self.assertIsNone(p.completed(directory, identity))

    def test_checkpoint_hash_mismatch_rejected(self):
        p, directory, identity, result = self.complete_fixture()
        result["checkpoint_sha256"] = "wrong"
        search.write_json(directory / "result.json", result)
        self.assertIsNone(p.completed(directory, identity))


if __name__ == "__main__":
    unittest.main()
