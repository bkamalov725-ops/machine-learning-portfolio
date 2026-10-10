"""Synthetic packaging, timer and baseline integration checks."""
import ast
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

import build_notebook as builder
import geo_v7_45_baselines as baselines


def source(notebook, identity):
    value = next(c for c in notebook["cells"] if c["id"] == identity)["source"]
    return "".join(value) if isinstance(value, list) else value


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.notebook = json.loads(builder.build(Path(cls.temp.name)/"fixed.ipynb").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_hashes_clean_cells_and_code(self):
        embedded = next(ast.literal_eval(node.value) for node in ast.parse(source(self.notebook,"embedded-code")).body
                        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                        and node.targets[0].id == "EMBEDDED_SOURCES")
        self.assertEqual(embedded, builder.embedded_sources())
        self.assertEqual(self.notebook["metadata"]["geo_final_sources"],
                         {name:hashlib.sha256(value.encode()).hexdigest() for name,value in embedded.items()})
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"],[])
                self.assertIsNone(cell["execution_count"])
                compile(source(self.notebook,cell["id"]),cell["id"],"exec")

    def test_explicit_phases_and_no_test_in_training_cell(self):
        self.assertEqual(self.notebook["metadata"]["geo_final_training_limit_minutes"],45)
        final = source(self.notebook,"final-training-timer")
        self.assertIn("experiment.run_final_train(pool)",final)
        self.assertNotIn("predict_final",final)
        self.assertNotIn("build_cache",final)
        self.assertIn("include_test=False",source(self.notebook,"prepare-labeled-only"))
        docs = "\n".join(c["source"] for c in self.notebook["cells"] if c["cell_type"]=="markdown")
        for phrase in ("не ко всему Run All", "MLP", "test", "FP32", "14:00 МСК", "минимум 3"):
            self.assertIn(phrase,docs)

    def run_timer(self,seconds=600,fail=False):
        with tempfile.TemporaryDirectory() as temporary:
            def train(pool):
                if fail:
                    raise RuntimeError("synthetic failed selected contributor")
                return [{"epochs_done":14}]
            times = iter([1000.,1000.+seconds])
            mono = iter([0.,seconds])
            clock = types.SimpleNamespace(time=lambda:next(times),monotonic=lambda:next(mono))
            namespace = dict(time=clock, RUN_DIR=Path(temporary), pool=object(),
                experiment=types.SimpleNamespace(run_final_train=train),
                traceback=types.SimpleNamespace(print_exc=lambda:None))
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                exec(source(self.notebook,"final-training-timer"),namespace)
            result = json.loads((Path(temporary)/"final_wallclock_notebook.json").read_text())
            return namespace,result

    def test_timer_records_real_wall_and_stops_before_inference(self):
        values,result = self.run_timer()
        self.assertTrue(values["FINAL_TIMER_VALID"])
        self.assertEqual(result["elapsed_minutes"],10)
        self.assertFalse(result["includes_test_prediction"])

    def test_overtime_and_training_errors_never_valid(self):
        values,_ = self.run_timer(2701)
        self.assertFalse(values["FINAL_TIMER_VALID"])
        values,_ = self.run_timer(fail=True)
        self.assertFalse(values["FINAL_TIMER_VALID"])

    def test_invalid_training_does_not_predict_or_link_stale_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"old.csv").write_text("synthetic previous CSV")
            calls=[]
            def report(pool,archive_status=None):
                (root/"report.txt").write_text("synthetic aggregate")
                return {}
            experiment=types.SimpleNamespace(predict_final=lambda p:calls.append("predict"),
                make_report=report,bundle=lambda p:{"created":True},SUBMISSION_NAME="old.csv",
                REPORT_NAME="report.txt",ARCHIVE_NAME="recovery.tar.gz")
            namespace=dict(experiment=experiment,RUN_DIR=root,pool=object(),FINAL_TIMER_VALID=False,
                           FileLink=str,display=lambda value:calls.append(value),traceback=types.SimpleNamespace(print_exc=lambda:None))
            with redirect_stdout(io.StringIO()):
                exec(source(self.notebook,"predict-and-save"),namespace)
            self.assertNotIn("predict",calls)
            self.assertNotIn(str(root/"old.csv"),calls)

    def test_aggregate_plots_run_on_synthetic_metrics_without_test(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"baselines").mkdir()
            hist=[dict(epoch=1,train_loss=1.,val_loss=.9,train_accuracy=.5,val_accuracy=.5)]
            for name in ("mlp","cnn"):
                (root/"baselines"/f"{name}_history.json").write_text(json.dumps(hist))
            pred=root/"synthetic_pred"
            pred.mkdir()
            (pred/"job.json").write_text(json.dumps({"tta":"none"}))
            records=[dict(key="r152",result_dir=str(pred))]
            (root/"screen_predictions.json").write_text(json.dumps(records))
            (root/"screen_model_metrics.json").write_text(json.dumps([dict(key="r152",accuracy=1.)]))
            experiment=types.SimpleNamespace(read_json=lambda p:json.loads(Path(p).read_text()),
                read_prediction=lambda r,ids,classes:{"probs":np.asarray([[.9,.1],[.1,.9]])},
                mixture=lambda v,w:v["r152"])
            namespace=dict(RUN_DIR=root,BASELINE_RESULTS=[dict(model="mlp",validation_accuracy=.5)],
                SELECTED_RECIPE={"selected":{"tta":"none","weights":{"r152":1.}}},experiment=experiment,
                pool=types.SimpleNamespace(val=pd.DataFrame({"id":["v","w"],"label":[0,1]}),classes=["a","b"]),
                Path=Path,display=lambda x:None)
            with mock.patch.object(plt,"show"),redirect_stdout(io.StringIO()):
                exec(source(self.notebook,"aggregate-comparison"),namespace)
            self.assertTrue((root/"screen_confusion_matrix.png").is_file())
            self.assertTrue((root/"baseline_learning_curves.png").is_file())
            plt.close("all")


class BaselineTests(unittest.TestCase):
    def test_baseline_models_output_and_seed_before_creation(self):
        torch.set_num_threads(1)
        for name in ("mlp","cnn"):
            torch.manual_seed(2)
            model=baselines.baseline_model(name,3).eval()
            with torch.inference_mode():
                self.assertEqual(tuple(model(torch.zeros(2,3,64,64)).shape),(2,3))
            torch.manual_seed(2)
            repeat=baselines.baseline_model(name,3)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),repeat.parameters())))

    def test_synthetic_baseline_train_only_complete_coverage_and_save(self):
        tables=dict(classes=["a","b"],train=pd.DataFrame({"id":["a","b","c"],"label":[0,1,0]}),
                    val=pd.DataFrame({"id":["v","w"],"label":[0,1]}),data_signature={"train.csv":"fake"})
        class TinyDataset:
            def __init__(self,tables,data_dir,split,transform,cache_dir=None):
                self.frame=tables[split]
            def __len__(self): return len(self.frame)
            def __getitem__(self,index): return torch.zeros(3,8,8),int(self.frame.iloc[index]["label"])
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(baselines.worker,"load_metadata",return_value=tables) as metadata, \
             mock.patch.object(baselines.worker,"ImageDataset",TinyDataset), \
             mock.patch.object(baselines,"baseline_model",side_effect=lambda n,c:nn.Sequential(nn.Flatten(),nn.Linear(192,c))):
            rows=baselines.run_baselines("unused",temporary,"unused",epochs=1,workers=0,allow_cpu=True)
            metadata.assert_called_once_with("unused",include_test=False)
            self.assertEqual(len(rows),2)
            self.assertTrue(all(row["fit_test_images"]==0 and row["outside_final_training_timer"] for row in rows))
            self.assertTrue((Path(temporary)/"baselines"/"mlp_fp32.pt").exists())
            self.assertTrue((Path(temporary)/"baselines"/"cnn_fp32.pt").exists())


if __name__ == "__main__":
    unittest.main()
