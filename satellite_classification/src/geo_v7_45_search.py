"""Team training controller: labeled selection, fresh <=45-minute final fit.

Only predict_final can access test. Screening is outside the final-fit timer.
Final checkpoints are FP32 last complete-epoch EMA, never best-on-val weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torchvision

import geo_v7_45_worker as worker

PROTOCOL = "v7_final45_labeled_only_1"
REPORT_NAME = "REPORT_V7_45_TO_SEND.txt"
SUBMISSION_NAME = "submission_geo_v7_45.csv"
ARCHIVE_NAME = "geo_v7_45_recovery.tar.gz"
CODE_FILES = ("geo_worker.py", "geo_search.py", "geo_v7_45_worker.py", "geo_v7_45_search.py", "geo_v7_45_baselines.py")
MODEL_SPECS = (("r152", "resnet152", 2026), ("r101", "resnet101", 2028), ("inc", "inception_v3", 2029))
TTAS = ("none", "d4")
FINAL_LIMIT_SECONDS = 45 * 60
TRAIN_STOP_SECONDS = 40 * 60
MIN_FINAL_EPOCHS = 3


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def probabilities(value, rows=None, classes=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not value.size or (rows is not None and len(value) != rows):
        raise ValueError("Invalid prediction row count")
    if classes is not None and value.shape[1] != classes:
        raise ValueError("Invalid prediction class count")
    if not np.isfinite(value).all() or (value < -1e-7).any() or (value > 1.000001).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(value.sum(1), 1, atol=2e-4, rtol=0):
        raise ValueError("Probability rows do not sum to one")
    return value


def mixture(values, weights):
    if not weights or len(weights) > 2 or any(v <= 0 for v in weights.values()) or abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("Recipe must contain one or two positive contributors summing to one")
    shapes = {np.asarray(values[k]).shape for k in weights}
    if len(shapes) != 1:
        raise ValueError("Contributor shapes differ")
    return probabilities(sum(probabilities(values[k]) * v for k, v in weights.items()))


def score(value, labels):
    labels = np.asarray(labels, dtype=np.int64)
    value = probabilities(value, len(labels))
    if labels.ndim != 1 or (labels < 0).any() or (labels >= value.shape[1]).any():
        raise ValueError("Invalid validation labels")
    correct = int((value.argmax(1) == labels).sum())
    return dict(correct=correct, count=len(labels), accuracy=correct / len(labels),
                loss=float(-np.log(np.clip(value[np.arange(len(labels)), labels], 1e-12, 1)).mean()))


def recipes():
    return [("r152", {"r152": 1.}), ("r101", {"r101": 1.}), ("inc", {"inc": 1.}),
            ("r152_inc_60_40", {"r152": .6, "inc": .4}),
            ("r152_inc_40_60", {"r152": .4, "inc": .6}),
            ("r152_r101_70_30", {"r152": .7, "r101": .3})]


def choose_recipe(predictions, labels):
    rows = []
    for tta in TTAS:
        for name, weights in recipes():
            rows.append(dict(score(mixture(predictions[tta], weights), labels), name=name + "_" + tta,
                             weights=weights, tta=tta, views=1 if tta == "none" else 8))
    rows.sort(key=lambda row: (-row["correct"], row["loss"], row["views"], row["name"]))
    return rows[0], rows


def stop_process(process):
    """Bounded termination of worker and Linux DataLoader process group."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
        process.wait(timeout=4)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=4)


class Pool:
    def __init__(self, data_dir, run_dir, code_dir, cache_dir, workers=2):
        self.data_dir, self.output_dir, self.code_dir, self.cache_dir = map(
            lambda p: Path(p).resolve(), (data_dir, run_dir, code_dir, cache_dir))
        for other in (self.data_dir, self.cache_dir):
            if other.is_relative_to(self.output_dir) or self.output_dir.is_relative_to(other):
                raise ValueError("Run output must be separate from data and image cache")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers, self.started = int(workers), time.time()
        # No test table or image is read during initialization or selection.
        metadata = worker.load_metadata(self.data_dir, include_test=False)
        self.train, self.val = metadata["train"], metadata["val"]
        self.classes = list(metadata["classes"])
        self.class_labels = list(range(len(self.classes)))
        self.data_signature = {name: digest(self.data_dir / name) for name in ("classes.csv", "train.csv", "val.csv")}
        self.source_hashes = {name: digest(self.code_dir / name) for name in CODE_FILES}
        self.configs = {key: worker.config_for_model(model, seed, trial_id="v7_45_" + key)
                        for key, model, seed in MODEL_SPECS}
        self.gpus = list(range(torch.cuda.device_count()))[:2]
        if not self.gpus:
            raise RuntimeError("Enable Kaggle GPU; CPU final training is not supported")
        self.failures, self.events = [], []
        self.environment = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
                                numpy=np.__version__, pandas=pd.__version__, cuda=torch.version.cuda,
                                cudnn=torch.backends.cudnn.version(),
                                gpu_names=[torch.cuda.get_device_name(gpu) for gpu in self.gpus])
        context = dict(protocol=PROTOCOL, data_signature=self.data_signature, source_hashes=self.source_hashes,
                       configs=self.configs, recipes=recipes(), tta=TTAS, final_limit_seconds=FINAL_LIMIT_SECONDS,
                       training_stop_seconds=TRAIN_STOP_SECONDS, final_reuse=False, no_test_fitting=True)
        context = json.loads(json.dumps(context))
        prior = self.output_dir / "run_context.json"
        if prior.exists() and read_json(prior) != context:
            raise RuntimeError("Existing RUN_DIR has different source/data/config. Use a new RUN_DIR.")
        write_json(prior, context)
        write_json(self.output_dir / "environment.json", self.environment)

    def event(self, kind, **details):
        row = dict(time_unix=time.time(), kind=kind, **details)
        self.events.append(row)
        write_json(self.output_dir / "events.json", self.events)
        print(f"[V7 final45] {kind}: {details}", flush=True)

    def identity(self, payload):
        keep = {k: v for k, v in payload.items() if k not in {"deadline_unix", "output_dir"}}
        keep.update(controller_protocol=PROTOCOL, source_hashes=self.source_hashes, data_signature=self.data_signature,
                    data_dir=str(self.data_dir), cache_dir=str(self.cache_dir), workers=self.workers)
        return json.loads(json.dumps(keep))

    def completed(self, directory, identity, prediction=False):
        try:
            if read_json(directory / "controller_identity.json") != identity:
                return None
            result = read_json(directory / "result.json")
            if result.get("config") != identity["config"] or not result.get("complete_protocol") or result.get("classes") != self.classes:
                return None
            if result.get("worker_protocol") != worker.PROTOCOL:
                return None
            job = read_json(directory / "job.json")
            if result.get("job_signature") != worker.job_signature(job) or result.get("phase") != identity["phase"]:
                return None
            if any(result.get("data_signature", {}).get(k) != v for k, v in self.data_signature.items()):
                return None
            expected_source = {k: self.source_hashes[k] for k in ("geo_worker.py", "geo_v7_45_worker.py")}
            if result.get("source_signature") != expected_source:
                return None
            if prediction:
                checkpoint = Path(identity["checkpoint"])
                prediction_file = directory / "probs.npz"
                if not prediction_file.is_file() or result.get("checkpoint_sha256") != digest(checkpoint):
                    return None
                if result.get("probs_sha256") != digest(prediction_file):
                    return None
            else:
                epochs = int(result.get("completed_epochs", result.get("epochs_done", 0)))
                if not result.get("eligible") or epochs < (MIN_FINAL_EPOCHS if identity["phase"] == "final" else int(identity["config"]["epochs"])):
                    return None
                if epochs > int(identity["config"]["epochs"]) or result.get("weight_kind") != "fixed_last_ema":
                    return None
                if result.get("train_images") != len(self.train) or result.get("fit_test_images") != 0:
                    return None
                if result.get("fit_val_images") != (len(self.val) if identity["phase"] == "final" else 0):
                    return None
                checkpoint = Path(result.get("checkpoint", directory / "best.pt")).resolve()
                if not checkpoint.is_relative_to(directory.resolve()):
                    return None
                if not checkpoint.is_file() or result.get("checkpoint_sha256") != digest(checkpoint):
                    return None
            return dict(result, key=identity["key"], phase=identity["phase"], config=identity["config"],
                        result_dir=str(directory), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                        controller_identity=fingerprint(identity))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def run(self, payloads, prediction=False, deadline=None, reuse=True, attempt_root=None):
        queue, done, active = [], [], []
        for payload in payloads:
            identity = self.identity(payload)
            name = payload["key"] + ("_" + payload["tta"] if prediction else "")
            parent = (Path(attempt_root) if attempt_root else self.output_dir / "screen") / ("predict" if prediction else "train") / name
            parent.mkdir(parents=True, exist_ok=True)
            previous = None
            if reuse:
                for directory in sorted(parent.glob("attempt_*")):
                    previous = self.completed(directory, identity, prediction)
                    if previous:
                        break
            if previous:
                done.append(previous)
                self.event("reuse_complete_screen", key=name)
            else:
                directory = parent / ("attempt_" + uuid.uuid4().hex[:12])
                queue.append((payload, identity, directory, name))
        try:
            while queue or active:
                now = time.time()
                if deadline is not None and now >= deadline:
                    for item in list(active):
                        running = item["process"].poll() is None
                        if running:
                            stop_process(item["process"])
                        item["log"].close()
                        result = self.completed(item["directory"], item["identity"], prediction) if running or item["process"].returncode == 0 else None
                        if result:
                            done.append(result)
                        else:
                            self.failures.append(dict(key=item["name"], reason="deadline_no_eligible_result"))
                    active.clear()
                    self.failures.extend(dict(key=name, reason="deadline_before_launch") for _, _, _, name in queue)
                    queue.clear()
                    break
                free = [gpu for gpu in self.gpus if gpu not in [item["gpu"] for item in active]]
                while free and queue:
                    payload, identity, directory, name = queue.pop(0)
                    gpu = free.pop(0)
                    directory.mkdir(parents=True)
                    write_json(directory / "controller_identity.json", identity)
                    job_deadline = deadline if deadline is not None else time.time() + (2 if prediction else 4) * 3600
                    job = dict(payload, data_dir=str(self.data_dir), output_dir=str(directory), cache_dir=str(self.cache_dir),
                               workers=self.workers, deadline_unix=job_deadline, allow_cpu=False)
                    write_json(directory / "job.json", job)
                    log = (directory / "worker.log").open("w", encoding="utf-8")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                               MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                    try:
                        process = subprocess.Popen([sys.executable, str(self.code_dir / "geo_v7_45_worker.py"),
                                                    "--job", str(directory / "job.json")],
                                                   stdout=log, stderr=subprocess.STDOUT, cwd=str(self.code_dir), env=env,
                                                   start_new_session=os.name == "posix")
                    except BaseException:
                        log.close()
                        raise
                    active.append(dict(process=process, log=log, directory=directory, identity=identity,
                                       name=name, gpu=gpu, deadline=job_deadline))
                    self.event("launch_prediction" if prediction else "launch_training", key=name, gpu=gpu)
                for item in list(active):
                    process = item["process"]
                    if process.poll() is None and time.time() < item["deadline"]:
                        continue
                    stopped = process.poll() is None
                    if stopped:
                        stop_process(process)
                    item["log"].close()
                    result = self.completed(item["directory"], item["identity"], prediction) if stopped or process.returncode == 0 else None
                    if result:
                        done.append(result)
                        self.event("worker_complete", key=item["name"], epochs=result.get("completed_epochs"))
                    else:
                        self.failures.append(dict(key=item["name"], reason="no_eligible_result", returncode=process.returncode,
                                                  timed_out=stopped, result_dir=str(item["directory"])))
                    active.remove(item)
                if active:
                    time.sleep(.25)
        finally:
            for item in active:
                stop_process(item["process"])
                item["log"].close()
            write_json(self.output_dir / "failures.json", self.failures)
        return done


def prediction_payload(record, split, tta):
    return dict(mode="predict", phase="predict", key=record["key"], config=record["config"],
                checkpoint=record["checkpoint"], checkpoint_sha256=digest(record["checkpoint"]), split=split, tta=tta)


def read_prediction(record, ids, classes):
    with np.load(Path(record["result_dir"]) / "probs.npz", allow_pickle=False) as archive:
        got_ids = archive["ids"].astype(str).tolist()
        got_classes = archive["classes"].astype(str).tolist()
        if got_ids != [str(x) for x in ids] or got_classes != list(classes):
            raise ValueError("ID/class order differs from the frozen prediction protocol")
        value = probabilities(archive["probs"], len(ids), len(classes))
        if "checkpoint_sha256" not in archive or str(archive["checkpoint_sha256"].item()) != record["checkpoint_sha256"]:
            raise ValueError("Prediction refers to different checkpoint bytes")
        return dict(probs=value, labels=archive["labels"].astype(np.int64).tolist())


def run_screen(pool):
    pool.event("cache_labeled_only_before_screen")
    worker.build_cache(pool.data_dir, pool.cache_dir, side=320, include_test=False)
    jobs = [dict(mode="train", phase="screen", key=key, config=config,
                 train_val_indices=[], holdout_val_indices=list(range(len(pool.val))))
            for key, config in pool.configs.items()]
    fits = pool.run(jobs)
    if len(fits) != len(jobs) or {x["key"] for x in fits} != set(pool.configs):
        raise RuntimeError("All three complete train-only screening fits are required; no test accessed")
    write_json(pool.output_dir / "screen_fits.json", fits)
    predictions = pool.run([prediction_payload(record, "val", tta) for record in fits for tta in TTAS], prediction=True)
    if len(predictions) != len(fits) * len(TTAS):
        raise RuntimeError("Validation inference incomplete; no recipe frozen and no test accessed")
    write_json(pool.output_dir / "screen_predictions.json", predictions)
    return fits


def select_recipe(pool):
    fits = read_json(pool.output_dir / "screen_fits.json")
    predictions = read_json(pool.output_dir / "screen_predictions.json")
    values = {tta: {} for tta in TTAS}
    individual = []
    for record in predictions:
        job = read_json(Path(record["result_dir"]) / "job.json")
        array = read_prediction(record, pool.val["id"].tolist(), pool.classes)
        if array["labels"] != pool.val["label"].tolist():
            raise ValueError("Validation labels and prediction rows are not aligned")
        values[job["tta"]][record["key"]] = array["probs"]
        individual.append(dict(key=record["key"], tta=job["tta"], **score(array["probs"], array["labels"])))
    selected, table = choose_recipe(values, pool.val["label"].to_numpy())
    recipe = dict(protocol=PROTOCOL, selected=selected, configs={key: pool.configs[key] for key in selected["weights"]},
                  data_signature=pool.data_signature, source_hashes=pool.source_hashes,
                  screen_checkpoint_hashes={row["key"]: row["checkpoint_sha256"] for row in fits},
                  final_policy=dict(max_epochs=14, minimum_epochs=MIN_FINAL_EPOCHS, stop_seconds=TRAIN_STOP_SECONDS,
                                    limit_seconds=FINAL_LIMIT_SECONDS, weight_kind="last complete epoch FP32 EMA",
                                    final_weight_reuse=False, val_repeat=2),
                  no_test_fitting=True, final_validation_accuracy_available=False,
                  validation_note="Train-only held-out val; reused 12-recipe selection, not a new independent test.")
    recipe["recipe_fingerprint"] = fingerprint(recipe)
    write_json(pool.output_dir / "validation_recipes.json", table)
    write_json(pool.output_dir / "screen_model_metrics.json", individual)
    write_json(pool.output_dir / "selected_recipe.json", recipe)
    pool.event("recipe_frozen_before_test", name=selected["name"], accuracy=selected["accuracy"])
    return recipe


def verify_frozen_recipe(pool):
    recipe = read_json(pool.output_dir / "selected_recipe.json")
    signature = recipe.pop("recipe_fingerprint")
    if fingerprint(recipe) != signature or recipe.get("source_hashes") != pool.source_hashes or recipe.get("data_signature") != pool.data_signature:
        raise RuntimeError("Frozen selection recipe was changed; rerun labeled screening")
    recipe["recipe_fingerprint"] = signature
    return recipe


def run_final_train(pool):
    recipe = verify_frozen_recipe(pool)
    selected = recipe["selected"]
    attempt = pool.output_dir / "final_attempts" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True)
    # Preserve old artifacts recoverably, but remove their misleading top-level
    # links before a fresh attempt so a failed run cannot look like a new CSV.
    previous = attempt / "previous_output"
    for name in (SUBMISSION_NAME, "submission_summary.json", "final_ensemble_probabilities.npz", "final_fits.json",
                 "frozen_final.json", "reproduce_final_recipe.json"):
        path = pool.output_dir / name
        if path.is_file():
            previous.mkdir(exist_ok=True)
            os.replace(path, previous / name)
    # Every call trains fresh weights. Neither completed screen nor prior final
    # results can replace this timed final fit on train+val.
    started_wall, started_monotonic = time.time(), time.monotonic()
    timer = dict(attempt=str(attempt), start_time=started_wall, end_time=None, elapsed_seconds=None,
                 elapsed_minutes=None, within_limit=False, training_performed=True, final_weight_reuse=False,
                 final_limit_seconds=FINAL_LIMIT_SECONDS, training_stop_seconds=TRAIN_STOP_SECONDS,
                 included="model initialization, process startup, all final training, EMA checkpoint saves and process teardown",
                 excluded="screening, validation model comparison and test prediction", status="running")
    write_json(pool.output_dir / "final_training_time.json", timer)
    write_json(attempt / "frozen_selection.json", recipe)
    fits, error = [], None
    try:
        jobs = [dict(mode="train", phase="final", key=key, config=recipe["configs"][key], min_epochs=MIN_FINAL_EPOCHS,
                     train_val_indices=list(range(len(pool.val))), holdout_val_indices=[])
                for key in selected["weights"]]
        fits = pool.run(jobs, deadline=started_wall + TRAIN_STOP_SECONDS, reuse=False, attempt_root=attempt)
        if len(fits) != len(jobs) or {row["key"] for row in fits} != set(selected["weights"]):
            raise RuntimeError("Not all selected final models reached three complete epochs; no partial/reweighted CSV is allowed")
    except BaseException as exc:
        error = exc
    finally:
        elapsed = time.monotonic() - started_monotonic
        timer.update(end_time=time.time(), elapsed_seconds=elapsed, elapsed_minutes=elapsed / 60,
                     within_limit=elapsed <= FINAL_LIMIT_SECONDS,
                     status="complete" if error is None and elapsed <= FINAL_LIMIT_SECONDS else "failed")
        write_json(pool.output_dir / "final_training_time.json", timer)
        write_json(attempt / "final_training_time.json", timer)
        print(f"Elapsed time: {timer['elapsed_minutes']:.2f} minutes (final training only)", flush=True)
    if error is not None:
        raise error
    if not timer["within_limit"]:
        raise RuntimeError("Final training exceeded 45 minutes. No CSV can be exported from this attempt.")
    frozen = dict(protocol=PROTOCOL, selection_fingerprint=recipe["recipe_fingerprint"], selected=selected,
                  final_training_time=timer, models=[dict(key=row["key"], config=row["config"], checkpoint=row["checkpoint"],
                    checkpoint_sha256=row["checkpoint_sha256"], completed_epochs=row.get("completed_epochs", row.get("epochs_done")),
                    weight_kind=row.get("weight_kind"), result_dir=row["result_dir"]) for row in fits],
                  actual_recipe_matches_validation_selected_weights=True, final_validation_accuracy=None)
    frozen["final_fingerprint"] = fingerprint(frozen)
    write_json(pool.output_dir / "final_fits.json", fits)
    write_json(pool.output_dir / "frozen_final.json", frozen)
    write_json(attempt / "frozen_final.json", frozen)
    reproduce = dict(protocol=PROTOCOL, actual_completed_epochs={row["key"]: row.get("completed_epochs", row.get("epochs_done")) for row in fits},
                     configs={row["key"]: row["config"] for row in fits}, selected=selected,
                     data_signature=pool.data_signature, source_hashes=pool.source_hashes, environment=pool.environment,
                     final_fingerprint=frozen["final_fingerprint"], final_limit_seconds=FINAL_LIMIT_SECONDS,
                     exact_inference_note="Use these archived FP32 checkpoint bytes for the exact submitted CSV; seeds alone do not promise bit-exact retraining.",
                     fixed_epoch_retraining_note="A reviewer can stop each refit after actual_completed_epochs rather than hardware-dependent stopping, while preserving the original planned 14-epoch learning-rate schedule and all configs. Do NOT change cfg.epochs to the shorter count: that changes the schedule. Include those fits in the same 45-minute timer.")
    write_json(pool.output_dir / "reproduce_final_recipe.json", reproduce)
    write_json(attempt / "reproduce_final_recipe.json", reproduce)
    pool.event("all_final_weights_frozen", elapsed_minutes=timer["elapsed_minutes"], actual_epochs={r["key"]: r["completed_epochs"] for r in fits})
    return fits


def verify_final(pool):
    recipe = verify_frozen_recipe(pool)
    final = read_json(pool.output_dir / "frozen_final.json")
    signature = final.pop("final_fingerprint")
    if fingerprint(final) != signature:
        raise RuntimeError("Frozen final model manifest was changed")
    final["final_fingerprint"] = signature
    timer = read_json(pool.output_dir / "final_training_time.json")
    if timer != final["final_training_time"] or timer.get("status") != "complete" or not timer.get("training_performed"):
        raise RuntimeError("Current final attempt did not complete; an older final manifest cannot be reused")
    if not timer.get("within_limit") or not 0 < float(timer["elapsed_seconds"]) <= FINAL_LIMIT_SECONDS:
        raise RuntimeError("Final training timing is not valid for the 45-minute requirement")
    if final["selection_fingerprint"] != recipe["recipe_fingerprint"] or final["selected"] != recipe["selected"]:
        raise RuntimeError("Final models differ from validation-frozen selection")
    if len(final["models"]) != len(recipe["selected"]["weights"]) or {row["key"] for row in final["models"]} != set(recipe["selected"]["weights"]):
        raise RuntimeError("Final contributor set differs; renormalizing a partial ensemble is forbidden")
    for row in final["models"]:
        if int(row["completed_epochs"]) < MIN_FINAL_EPOCHS or digest(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise RuntimeError("Final checkpoint identity or complete-epoch count changed")
    return recipe, final


def test_metadata(pool):
    """Prediction-only table access; no test stats, labels, balance or fitting."""
    table = pd.read_csv(pool.data_dir / "test.csv", dtype={"id": str}, keep_default_na=False)
    if "id" not in table or not len(table) or table["id"].duplicated().any():
        raise ValueError("test.csv needs each nonempty image id exactly once")
    ids = table["id"].tolist()
    if any(not item.strip() for item in ids):
        raise ValueError("Empty test image ID")
    template = pool.data_dir / "sample_submission_example.csv"
    if template.exists():
        sample = pd.read_csv(template, dtype=str, keep_default_na=False)
        if len(sample.columns) != 2 or sample.columns[0] != "id" or sample.columns[1] not in {"label", "target"}:
            raise ValueError("Sample submission columns must be id,label or id,target")
        order, target = sample["id"].tolist(), str(sample.columns[1])
        if len(order) != len(ids) or len(set(order)) != len(order) or set(order) != set(ids):
            raise ValueError("Submission template IDs do not equal test IDs")
    else:
        order, target = ids, "label"
    return ids, order, target


def predict_final(pool):
    # All guard checks occur BEFORE test.csv, test image cache or test pixels.
    recipe, frozen = verify_final(pool)
    ids, order, target = test_metadata(pool)
    pool.event("test_prediction_only_after_all_final_weights_frozen")
    worker.build_cache(pool.data_dir, pool.cache_dir, side=320, include_test=True)
    jobs = [prediction_payload(record, "test", frozen["selected"]["tta"]) for record in frozen["models"]]
    root = Path(frozen["final_training_time"]["attempt"]) / "test_inference"
    records = pool.run(jobs, prediction=True, reuse=False, attempt_root=root)
    if len(records) != len(jobs) or {row["key"] for row in records} != set(frozen["selected"]["weights"]):
        raise RuntimeError("Incomplete final test inference; no partial submission is generated")
    values = {row["key"]: read_prediction(row, ids, pool.classes)["probs"] for row in records}
    value = mixture(values, frozen["selected"]["weights"])
    predicted = value.argmax(1)
    position = {item: index for index, item in enumerate(ids)}
    frame = pd.DataFrame({"id": order, target: [int(predicted[position[item]]) for item in order]})
    destination = pool.output_dir / SUBMISSION_NAME
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    np.savez_compressed(pool.output_dir / "final_ensemble_probabilities.npz", ids=np.asarray(ids, dtype=str),
                        probs=value.astype(np.float32), classes=np.asarray(pool.classes, dtype=str))
    summary = dict(filename=SUBMISSION_NAME, sha256=digest(destination), rows=len(frame), target_column=target,
                   selection_fingerprint=recipe["recipe_fingerprint"], final_fingerprint=frozen["final_fingerprint"],
                   weights=frozen["selected"]["weights"], tta=frozen["selected"]["tta"],
                   actual_epochs={row["key"]: row["completed_epochs"] for row in frozen["models"]},
                   checkpoint_sha256={row["key"]: row["checkpoint_sha256"] for row in frozen["models"]},
                   final_training_minutes=frozen["final_training_time"]["elapsed_minutes"], test_only_predicted=True,
                   no_test_class_balance_adjustment=True, no_test_pseudo_labels=True, final_validation_accuracy=None)
    write_json(pool.output_dir / "submission_summary.json", summary)
    write_json(root / "submission_summary.json", summary)
    pool.event("final_csv_complete", rows=len(frame), final_training_minutes=summary["final_training_minutes"])
    return summary


def make_report(pool, archive_status=None):
    def optional(name, default=None):
        path = pool.output_dir / name
        return read_json(path) if path.is_file() else default
    timer = optional("final_training_time.json")
    submission = optional("submission_summary.json")
    final = optional("frozen_final.json")
    # A failed new timed attempt must not be reported as an older valid CSV.
    current = bool(timer and final and timer == final.get("final_training_time") and timer.get("status") == "complete")
    if not current or (submission and final and submission.get("final_fingerprint") != final.get("final_fingerprint")):
        submission = None
    baseline = optional("baselines/results.json")
    report = dict(protocol=PROTOCOL, all_test_labels_unknown=True, public_score=None,
                  train_and_val_used_for_final_training=True, test_only_used_for_prediction=True,
                  no_test_stats_or_fitting=True, no_test_pseudo_labels=True, no_test_class_balance_adjustment=True,
                  selection=optional("selected_recipe.json"), screen_model_metrics=optional("screen_model_metrics.json", []),
                  validation_recipes=optional("validation_recipes.json", []), baseline_comparison=baseline,
                  validation_note="Train-only fits evaluated on held-out supplied val; reused recipe selection is not an independent test.",
                  final_validation_accuracy=None, final_training_time=timer, final_attempt_current=current,
                  actual_final_models=[{k: row.get(k) for k in ("key", "config", "completed_epochs", "checkpoint_sha256", "weight_kind")}
                                       for row in final.get("models", [])] if current else [],
                  submission=submission, failures=pool.failures, environment=pool.environment,
                  total_run_minutes=(time.time() - pool.started) / 60, archive=archive_status,
                  timing_note="45-minute limit applies only to fresh final training, including initialization/save/process teardown, not screening or test prediction.")
    write_json(pool.output_dir / "report_v7_45.json", report)
    text = "GEO V7 FINAL45: AGGREGATE REPORT\n\nTrain-only screening held val out; final training used permitted train+val.\nTest only predicted after recipe and all final weights were frozen.\nNo final-model validation accuracy is claimed. No test fitting, pseudolabels or class-prior correction.\n\n"
    (pool.output_dir / REPORT_NAME).write_text(text + json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def bundle(pool):
    """Private complete replay package; original images/uint8 caches excluded."""
    root = pool.output_dir.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".pt", ".json", ".csv", ".npz", ".txt", ".log", ".py"}:
            if path.name != "recovery_manifest.json" and not path.resolve().is_relative_to(pool.data_dir) and not path.resolve().is_relative_to(pool.cache_dir):
                files.append(("run/" + path.relative_to(root).as_posix(), path))
    files += [("code/" + name, pool.code_dir / name) for name in CODE_FILES]
    manifest = dict(protocol=PROTOCOL, files={name: dict(sha256=digest(path), size=path.stat().st_size) for name, path in files},
                    original_images_included=False, image_caches_included=False, includes_fp32_training_checkpoints=True)
    write_json(root / "recovery_manifest.json", manifest)
    destination = root / ARCHIVE_NAME
    temporary = destination.with_name(destination.name + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        archive.add(root / "recovery_manifest.json", arcname="manifest.json", recursive=False)
        for name, path in files:
            archive.add(path, arcname=name, recursive=False)
    os.replace(temporary, destination)
    result = dict(created=True, filename=ARCHIVE_NAME, size_bytes=destination.stat().st_size, sha256=digest(destination),
                  files=len(files), no_original_images=True, no_image_caches=True)
    write_json(root / "archive_status.json", result)
    return result
