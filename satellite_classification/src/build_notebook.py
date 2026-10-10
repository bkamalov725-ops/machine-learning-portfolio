"""Self-contained rules-aware replacement; never edits the user's source notebook."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import textwrap

from notebook_cells import code, markdown

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "notebooks" / "solution_kaggle.ipynb"
SOURCE_FILES = ("geo_worker.py", "geo_search.py", "geo_v7_45_worker.py", "geo_v7_45_search.py", "geo_v7_45_baselines.py")


def embedded_sources():
    result = {name: (HERE / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    for name, source in result.items():
        ast.parse(source, filename=name)
    return result


def build(target=None):
    sources = embedded_sources()
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()}
    cells = [
        markdown("intro", """
            # GEO V7: исправленная версия, финальное обучение не более 45 минут

            **45 минут относятся к обучению финального решения, а не ко всему Run All.**
            Предварительное сравнение MLP/CNN и предобученных моделей идёт отдельно.
            Полный первый запуск может занять больше 45 минут. Не выдаём его время
            за время одного финального обучения.

            Новый финальный этап каждый раз обучается заново. Его таймер включает
            запуск процессов, загрузку моделей, обучение, сохранение FP32 EMA и
            завершение процессов. До 45 минут есть запас: обучение ограничено
            меньшим внутренним бюджетом, зависшие процессы останавливает родитель.
            Нет фиктивного нулевого времени из-за готовых result.json.

            **Данные:** предварительные модели — только train, оценка на всём val.
            Финальные модели — train + val (val повторяется дважды), как разрешено
            для командного этапа. Это не индивидуальный checkpoint.

            **Test используется только в последней ячейке получения предсказаний.**
            Не анализируем его дефекты/распределение, не строим псевдометки,
            не подбираем по нему веса, аугментации или BatchNorm. Полностью исключить
            test и одновременно создать submission невозможно: для CSV нужны изображения.

            Не импортируются готовые модели или прогнозы других команд. Архитектуры
            и аугментации — разрешённые torchvision; веса только ImageNet.
        """),
        markdown("start", """
            ## Запуск

            1. Kaggle Input: **официальный полный датасет**, старые архивы не требуются.
            2. **T4 ×2, Internet ON**, затем Save Version → Run All.
            3. Для защиты на уже подготовленной сессии можно повторить отдельно
               ячейку **«Финальное обучение и официальный таймер»**, затем предсказания.
               Финальное обучение при этом действительно запускается заново.

            Предварительные эксперименты не включаются в лимит финального обучения,
            но остаются в коде/отчёте. Срок из присланных правил — **10 октября 2026,
            14:00 МСК**; заранее оставь время на CSV и Moodle.
        """),
        code("settings", """
            import os, time, hashlib, importlib, sys, traceback
            from pathlib import Path
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            NOTEBOOK_STARTED = time.time()

            DATA_DIR_OVERRIDE = None
            RUN_DIR = Path("/kaggle/working/geo_v7_45")
            CODE_DIR = Path("/kaggle/working/geo_v7_45_code")
            CACHE_DIR = Path("/tmp/geo_v7_45_images")
            BASELINE_EPOCHS = 8

            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Включи GPU в Kaggle. CPU не подходит для защиты с этим лимитом.")
            print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
            if torch.cuda.device_count() < 2:
                print("Только один GPU: времени меньше, финальное обучение может не завершиться.")
        """),
    ]
    embedded = "# Только собственный исполняемый код. Изображений, CSV и весов во вложении нет.\n"
    embedded += "EMBEDDED_SOURCES = " + repr(sources) + "\nEMBEDDED_HASHES = " + repr(hashes) + "\n"
    embedded += textwrap.dedent("""
        CODE_DIR.mkdir(parents=True, exist_ok=True)
        for name, source in EMBEDDED_SOURCES.items():
            if hashlib.sha256(source.encode()).hexdigest() != EMBEDDED_HASHES[name]:
                raise RuntimeError("Изменился встроенный исходник: " + name)
            (CODE_DIR / name).write_text(source, encoding="utf-8")
        if str(CODE_DIR) not in sys.path:
            sys.path.insert(0, str(CODE_DIR))
        for name in EMBEDDED_SOURCES:
            sys.modules.pop(Path(name).stem, None)
        importlib.invalidate_caches()
        import geo_search
        import geo_v7_45_worker as worker
        import geo_v7_45_search as experiment
        import geo_v7_45_baselines as baselines
        from IPython.display import display, FileLink
    """)
    cells.append(code("embedded-code", embedded))
    cells += [
        code("prepare-labeled-only", """
            DATA_DIR = geo_search.prepare_data(DATA_DIR_OVERRIDE)
            pool = experiment.Pool(DATA_DIR, RUN_DIR, CODE_DIR, CACHE_DIR)
            worker.build_cache(DATA_DIR, CACHE_DIR, include_test=False)
            print("Подготовлены только train и val; test не открыт.")
        """),
        markdown("baselines-note", """
            ## Обязательное сравнение MLP и собственной CNN

            Обе модели обучаются только на полном train, val используется только
            для оценки. Seed установлен **до создания** каждой модели. Сохраняются
            FP32-веса, loss/accuracy и длительность. Это предварительные эксперименты,
            не финальное обучение, и эти модели не входят в итоговый ансамбль.
        """),
        code("baselines", """
            BASELINE_RESULTS = baselines.run_baselines(
                DATA_DIR, RUN_DIR, CACHE_DIR, epochs=BASELINE_EPOCHS
            )
            for row in BASELINE_RESULTS:
                print(row)
        """),
        markdown("screen-note", """
            ## Предварительное обучение и выбор рецепта

            ResNet152, ResNet101 и Inception v3: только train, до 14 фиксированных
            эпох, последняя EMA, без выбора лучшей эпохи по val. Ранние ResNet-блоки
            заморожены вместе с BN-статистиками. Аугментации стандартными torchvision
            transforms, без собственной симуляции JPEG/DCT.

            На полном val сравнивается ограниченный заранее заданный набор ансамблей
            максимум из двух моделей и none/D4. Основная метрика — Accuracy, затем
            log-loss. Рецепт фиксируется до доступа к test. Его оценка на уже использованном
            для выбора val не является новым независимым тестом.
        """),
        code("screen-and-freeze", """
            SCREEN_FITS = experiment.run_screen(pool)
            SELECTED_RECIPE = experiment.select_recipe(pool)
            print("Зафиксированный рецепт:", SELECTED_RECIPE)
            experiment.make_report(pool)
        """),
        code("aggregate-comparison", """
            import json, numpy as np, pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            print("MLP/CNN — train-only оценка:")
            display(pd.DataFrame(BASELINE_RESULTS))
            print("Предобученные модели — train-only оценка:")
            display(pd.DataFrame(experiment.read_json(RUN_DIR / "screen_model_metrics.json")))

            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            for row, name in enumerate(("mlp", "cnn")):
                hist = pd.DataFrame(experiment.read_json(RUN_DIR / "baselines" / f"{name}_history.json"))
                axes[row, 0].plot(hist.epoch, hist.train_loss, label="train")
                axes[row, 0].plot(hist.epoch, hist.val_loss, label="val")
                axes[row, 0].set_title(name + ": loss")
                axes[row, 1].plot(hist.epoch, hist.train_accuracy, label="train")
                axes[row, 1].plot(hist.epoch, hist.val_accuracy, label="val")
                axes[row, 1].set_title(name + ": accuracy")
            for axis in axes.flat:
                axis.set_xlabel("epoch"); axis.legend(); axis.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(RUN_DIR / "baseline_learning_curves.png", dpi=120)
            plt.show()

            selected = SELECTED_RECIPE["selected"]
            values = {}
            for record in experiment.read_json(RUN_DIR / "screen_predictions.json"):
                job = experiment.read_json(Path(record["result_dir"]) / "job.json")
                if job["tta"] == selected["tta"] and record["key"] in selected["weights"]:
                    values[record["key"]] = experiment.read_prediction(
                        record, pool.val["id"].tolist(), pool.classes
                    )["probs"]
            guess = experiment.mixture(values, selected["weights"]).argmax(1)
            truth = pool.val["label"].to_numpy()
            cm = np.zeros((len(pool.classes), len(pool.classes)), dtype=np.int64)
            np.add.at(cm, (truth, guess), 1)
            pd.DataFrame(cm, index=pool.classes, columns=pool.classes).to_csv(RUN_DIR / "screen_confusion_matrix.csv")
            fig, axis = plt.subplots(figsize=(12, 10))
            ConfusionMatrixDisplay(cm, display_labels=pool.classes).plot(ax=axis, xticks_rotation=90, colorbar=False)
            axis.set_title("Train-only screening ensemble: reused held-out val")
            fig.tight_layout()
            fig.savefig(RUN_DIR / "screen_confusion_matrix.png", dpi=120)
            plt.show()
            # Показываем только агрегаты. Ни test, ни изображения здесь не открываются.
        """),
        markdown("final-note", """
            ## Финальное обучение и официальный таймер

            Эта ячейка каждый раз запускает **новое** обучение выбранных моделей на
            train + val. Старые финальные результаты не переиспользуются для таймера.
            Максимум 14 эпох; при ограничении бюджета сохраняется только последняя
            **полностью завершённая** эпоха, минимум 3. Реальное количество эпох
            и причина остановки записываются. Это обучение по бюджету, а не обещание,
            что любая остановленная модель прошла все 14 эпох.

            Все выбранные модели обязательны. Если одна не достигла минимального
            протокола или произошла ошибка, CSV не выдаётся за успешное решение.
            У завершённых train+val моделей **нет независимой validation accuracy**.
            Предварительную val accuracy им не приписываем.

            Для точного восстановления CSV нужны сохранённые FP32-веса, рецепт и
            совместимые версии библиотек: одних seeds недостаточно, особенно при
            остановке по времени. Сохрани все результаты приватно внутри команды.
        """),
        code("final-training-timer", """
            import json
            FINAL_TRAIN_ERROR = None
            FINAL_FITS = None
            start_time = time.time()
            start_monotonic = time.monotonic()
            try:
                FINAL_FITS = experiment.run_final_train(pool)
            except (Exception, KeyboardInterrupt) as error:
                FINAL_TRAIN_ERROR = f"{type(error).__name__}: {error}"
                print("Финальное обучение не завершено:", FINAL_TRAIN_ERROR)
                traceback.print_exc()
            finally:
                end_time = time.time()
                elapsed_minutes = (time.monotonic() - start_monotonic) / 60
                print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
                FINAL_TIMER_VALID = elapsed_minutes <= 45 and FINAL_TRAIN_ERROR is None
                (RUN_DIR / "final_wallclock_notebook.json").write_text(json.dumps({
                    "start_time": start_time, "end_time": end_time,
                    "elapsed_minutes": elapsed_minutes, "limit_minutes": 45,
                    "valid": FINAL_TIMER_VALID, "error": FINAL_TRAIN_ERROR,
                    "includes_startup_training_save_cleanup": True,
                    "includes_test_prediction": False,
                }, indent=2), encoding="utf-8")
            if not FINAL_TIMER_VALID:
                print("Нет успешного финального решения в лимите; test-предсказание будет пропущено.")
        """),
        markdown("predict-note", """
            ## Единственное использование test: итоговый CSV

            Предсказание — **после остановки таймера обучения**, только из зафиксированных
            FP32-checkpoints. Здесь нет обновления модели, BN-статистик, калибровки,
            поиска весов или балансировки классов test. Шаблон submission определяет
            названия столбцов и порядок IDs. Если обучение не завершено, test не читается.
        """),
        code("predict-and-save", """
            ARCHIVE_STATUS = "not_attempted"
            PREDICTION_VALID = False
            try:
                if FINAL_TIMER_VALID:
                    SUBMISSION = experiment.predict_final(pool)
                    PREDICTION_VALID = True
                    print("Итоговый CSV:", SUBMISSION)
                else:
                    print("Test не открыт: отсутствует корректное финальное обучение.")
            except (Exception, KeyboardInterrupt) as error:
                print("Ошибка предсказания:", type(error).__name__, str(error))
                traceback.print_exc()
            finally:
                try:
                    experiment.make_report(pool, archive_status="pending")
                    ARCHIVE_STATUS = experiment.bundle(pool)
                except (Exception, KeyboardInterrupt) as error:
                    ARCHIVE_STATUS = f"failed: {type(error).__name__}: {error}"
                    print("Сохрани папку outputs с весами целиком, если recovery не создался.")
                experiment.make_report(pool, archive_status=ARCHIVE_STATUS)
            print((RUN_DIR / experiment.REPORT_NAME).read_text(encoding="utf-8"))
            for name in (experiment.SUBMISSION_NAME, experiment.REPORT_NAME, experiment.ARCHIVE_NAME):
                path = RUN_DIR / name
                if path.is_file() and (name != experiment.SUBMISSION_NAME or PREDICTION_VALID):
                    display(FileLink(str(path)))
        """),
        markdown("results", """
            ## Что сохранить

            Папка **`/kaggle/working/geo_v7_45`**:

            - `submission_geo_v7_45.csv` — основной CSV, только после успешного финального обучения.
            - `REPORT_V7_45_TO_SEND.txt` — агрегированный отчёт, его можно прислать в чат.
            - `geo_v7_45_recovery.tar.gz` — веса/код/рецепты, сохранить приватно.
            - `final_wallclock_notebook.json` — замер финального обучения.

            Исходные изображения и uint8-кэш не включаются в recovery; не отправляй
            в ИИ-чат изображения, исходные таблицы и индивидуальные прогнозы.
            Для Moodle приложи notebook, все шаги решения, сравнение MLP/CNN/pretrained,
            фактические эпохи/версии/seeds и точное обозначение выбранного submission.
            Не публикуй приватные материалы и не передавай их другим командам.

            Этот код ограничивает время вычислений, но не обещает заранее public/private
            score или то, что Kaggle запустит сессию без очереди и внешних сбоев.
        """),
    ]
    notebook = dict(cells=cells, metadata=dict(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
                    language_info={"name": "python", "version": "3.12"},
                    geo_final_training_limit_minutes=45, geo_final_sources=hashes,
                    geo_test_policy="prediction_only_after_final_weights_frozen"), nbformat=4, nbformat_minor=5)
    destination = Path(target) if target is not None else TARGET
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build())
