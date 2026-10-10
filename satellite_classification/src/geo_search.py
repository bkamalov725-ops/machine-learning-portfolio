"""Independent-GPU experiment controller. No competition data is bundled.

The notebook calls run_pipeline() on Kaggle. All fits use train only; the broad
search uses a synthetic-corruption inner holdout, not the official validation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torchvision

BASE_SEED = 20261003


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stop_process(process):
    """Kaggle/Linux: terminate the whole trial, including DataLoader children."""
    try:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
        process.wait(timeout=15)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def find_data_dirs(root):
    root = Path(root)
    if not root.exists():
        return []
    required = ('train.csv', 'val.csv', 'test.csv', 'classes.csv')
    return sorted({p.parent.resolve() for p in root.rglob('train.csv')
                   if all((p.parent / name).is_file() for name in required)
                   and all((p.parent / name).is_dir() for name in ('train', 'val', 'test'))})


def prepare_data(explicit=None, input_root='/kaggle/input', working='/kaggle/working'):
    """Prefer explicit path; otherwise discover one dataset or unpack one ZIP."""
    if explicit:
        target = Path(explicit).resolve()
        if target not in find_data_dirs(target):
            raise FileNotFoundError(f'Неполный набор по DATA_DIR: {target}')
        return target
    roots = [Path(input_root), Path(working)]
    found = sorted(set(p for root in roots for p in find_data_dirs(root)))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError('Несколько наборов; укажи DATA_DIR явно: ' + str(found))
    archive_name = 'week-5-competition-geo-satellites-classification.zip'
    archives = sorted({p.resolve() for root in roots if root.exists()
                       for p in root.rglob(archive_name) if p.is_file()})
    if len(archives) != 1:
        raise FileNotFoundError('Добавь данные Kaggle или укажи DATA_DIR. Архивы: ' + str(archives))
    destination = Path(working) / 'geo_satellites_data'
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archives[0]) as archive:
        for item in archive.infolist():
            if not (destination / item.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Небезопасный путь в ZIP: ' + item.filename)
        archive.extractall(destination)
    found = find_data_dirs(destination)
    if len(found) != 1:
        raise RuntimeError('Проверь содержимое распакованного архива: ' + str(found))
    return found[0]


def screen_plan(budget='extended'):
    base = dict(seed=BASE_SEED, epochs=8, batch_size=32, lr=1e-3,
                weight_decay=1e-3, dropout=0.3, augment='robust', unfreeze='layer3',
                optimizer='adamw', label_smoothing=0.05, size=160)
    variants = [
        ('mlp', 'mlp', dict(size=32, epochs=16, batch_size=64, dropout=0.2,
                           augment='standard', label_smoothing=0.0)),
        ('cnn_plain', 'cnn_plain', dict(size=96, epochs=16, augment='standard', dropout=0.0)),
        ('cnn_bn', 'cnn_bn', dict(size=96, epochs=16, augment='standard', dropout=0.0)),
        ('cnn_gap_standard', 'cnn_gap', dict(size=128, epochs=20, augment='standard')),
        ('cnn_gap_robust', 'cnn_gap', dict(size=128, epochs=20)),
        ('r18_standard', 'resnet18', dict(augment='standard')),
        ('r18_robust', 'resnet18', dict()),
        ('r34_robust', 'resnet34', dict(lr=7e-4)),
        ('r50_robust', 'resnet50', dict(batch_size=24)),
    ]
    if budget == 'extended':
        variants += [
            ('r18_layer4', 'resnet18', dict(unfreeze='layer4')),
            ('r34_all', 'resnet34', dict(unfreeze='all', lr=5e-4)),
        ]
    elif budget != 'moderate':
        raise ValueError('budget must be moderate or extended')
    return [dict(base, **{'trial_id': name, 'model': model, **changes})
            for name, model, changes in variants]


def ranked_unique(results, count, predicate=lambda r: True):
    picked, seen = [], set()
    for result in sorted(results, key=lambda r: (-r['selection_metric'], r['trial_id'])):
        family = result['config']['model']
        if predicate(result) and family not in seen:
            picked.append(result)
            seen.add(family)
            if len(picked) == count:
                break
    return picked


def refinement_plan(results, budget):
    leaders = ranked_unique(results, 2, lambda r: r['config']['model'].startswith('resnet'))
    configs = []
    for rank, leader in enumerate(leaders):
        original = leader['config']
        changes = [('lr_half', dict(lr=original['lr'] / 2)),
                   ('lr_double', dict(lr=original['lr'] * 2))]
        if budget == 'extended':
            changes += [('sgd', dict(optimizer='sgd', lr=0.02, weight_decay=1e-4))]
        for name, change in changes:
            configs.append(dict(original, **{'trial_id': f'refine_{rank}_{name}', **change}))
    return configs


def final_plan(results, budget):
    chosen = ranked_unique(results, 2, lambda r: r['config']['model'].startswith('resnet'))
    chosen += ranked_unique(results, 1, lambda r: r['config']['model'].startswith('cnn'))
    chosen += ranked_unique(results, 1, lambda r: r['config']['model'] == 'mlp')
    configs = []
    for leader in chosen:
        cfg = dict(leader['config'])
        cfg['trial_id'] = 'final_' + cfg['trial_id']
        if cfg['model'].startswith('resnet'):
            cfg.update(size=224, epochs=24 if budget == 'extended' else 16,
                       batch_size=24 if cfg['model'] == 'resnet50' else 32)
        else:
            cfg['epochs'] = 45 if budget == 'extended' else 30
        configs.append(cfg)
    return configs


class GPUPool:
    """One child process per GPU; no DataParallel and no shared model state."""
    def __init__(self, data_dir, output_dir, workers=2):
        self.data_dir = str(Path(data_dir).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.worker = Path(__file__).with_name('geo_worker.py').resolve()
        self.workers = workers
        count = torch.cuda.device_count()
        if count < 1:
            raise RuntimeError('Включи GPU в Kaggle Settings → Accelerator. CPU-поиск отключён.')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        self.gpus = visible.split(',')[:min(count, 2)] if visible else [str(i) for i in range(min(count, 2))]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        context = {
            'data_dir': self.data_dir,
            'csv_hashes': {name: digest(Path(data_dir) / name) for name in
                           ('train.csv', 'val.csv', 'test.csv', 'classes.csv', 'sample_submission_example.csv')
                           if (Path(data_dir) / name).is_file()},
            'worker_sha256': digest(self.worker), 'controller_sha256': digest(__file__),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(len(self.gpus))],
            'workers': workers,
        }
        manifest = self.output_dir / 'run_context.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != context:
            raise RuntimeError('Изменились данные, код или окружение. Выбери новый RUN_NAME.')
        write_json(manifest, context)
        print('GPU для независимых запусков:', self.gpus, flush=True)

    def run(self, stage, jobs):
        pending, completed = [], []
        for identity, payload in jobs:
            base = self.output_dir / stage / identity
            job_signature = dict(payload, data_dir=self.data_dir, workers=self.workers, allow_cpu=False)
            signature_file = base / 'job_signature.json'
            if signature_file.exists():
                old = json.loads(signature_file.read_text(encoding='utf-8'))
                if old != job_signature:
                    raise RuntimeError(f'Настройки {identity} изменились: выбери другой RUN_NAME.')
            write_json(signature_file, job_signature)
            done = base / 'completed.json'
            if done.exists():
                previous = json.loads(done.read_text(encoding='utf-8'))
                if payload['phase'] != 'predict' and not (Path(previous['_directory']) / 'best.pt').exists():
                    raise FileNotFoundError('Checkpoint удалён: ' + previous['_directory'])
                if payload['phase'] != 'predict' and digest(Path(previous['_directory']) / 'best.pt') != previous.get('_checkpoint_sha256'):
                    raise RuntimeError('Checkpoint изменён после обучения: ' + previous['_directory'])
                if payload['phase'] == 'predict' and not Path(previous['_prefix'] + '.npz').exists():
                    raise FileNotFoundError('Прогнозы удалены: ' + previous['_prefix'])
                if payload['phase'] == 'predict' and digest(previous['_prefix'] + '.npz') != previous.get('_npz_sha256'):
                    raise RuntimeError('Сохранённые прогнозы изменены: ' + previous['_prefix'])
                completed.append(previous)
                print(f'[{stage}] reuse {identity}', flush=True)
                continue
            attempt = 1
            while (base / f'attempt_{attempt:03d}').exists():
                attempt += 1
            directory = base / f'attempt_{attempt:03d}'
            directory.mkdir(parents=True)
            job = dict(job_signature, output_dir=str(directory))
            if payload['phase'] == 'predict':
                job['prefix'] = str(directory / 'predictions')
            path = directory / 'job.json'
            write_json(path, job)
            pending.append((identity, job, path, done))
        active = {}
        last_status = time.monotonic()
        try:
            while pending or active:
                for gpu in self.gpus:
                    if gpu in active or not pending:
                        continue
                    identity, job, path, done = pending.pop(0)
                    logfile = Path(job['output_dir']) / 'stdout.log'
                    handle = logfile.open('w', encoding='utf-8')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1',
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                               CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    try:
                        process = subprocess.Popen([sys.executable, str(self.worker), '--job', str(path)],
                                                   stdout=handle, stderr=subprocess.STDOUT, env=env,
                                                   start_new_session=(os.name == 'posix'))
                    except BaseException:
                        handle.close()
                        raise
                    active[gpu] = (process, handle, identity, job, done, logfile)
                    print(f'[{stage}] GPU {gpu}: {identity}; log={logfile}', flush=True)
                for gpu in list(active):
                    process, handle, identity, job, done, logfile = active[gpu]
                    status = process.poll()
                    if status is None:
                        continue
                    handle.close()
                    del active[gpu]
                    if status:
                        stop_process(process)
                        tail = '\n'.join(logfile.read_text(encoding='utf-8', errors='replace').splitlines()[-35:])
                        raise RuntimeError(f'{identity} завершился с кодом {status}:\n{tail}')
                    directory = Path(job['output_dir'])
                    if job['phase'] == 'predict':
                        result = json.loads(Path(job['prefix'] + '.json').read_text(encoding='utf-8'))
                        result['_prefix'] = job['prefix']
                        result['_npz_sha256'] = digest(job['prefix'] + '.npz')
                    else:
                        result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
                        result['_checkpoint_sha256'] = digest(directory / 'best.pt')
                        if result.get('phase') != job['phase'] or result.get('config') != job['config'] or result.get('trial_id') != job['config']['trial_id'] or not np.isfinite(result.get('selection_metric', np.nan)):
                            raise ValueError('Неверный результат worker: ' + identity)
                    if job['phase'] == 'predict' and (result.get('split') != job['split'] or result.get('tta') != job['tta'] or result.get('checkpoint_sha256') != job['checkpoint_sha256']):
                        raise ValueError('Неверные метаданные предсказаний: ' + identity)
                    result.update(_directory=str(directory), _identity=identity)
                    write_json(done, result)
                    completed.append(result)
                    print(f'[{stage}] DONE {identity}: {result.get("selection_metric", result.get("accuracy", "OK"))}', flush=True)
                if time.monotonic() - last_status > 60:
                    for gpu, (_, _, identity, _, _, logfile) in active.items():
                        lines = logfile.read_text(encoding='utf-8', errors='replace').splitlines()
                        print(f'GPU {gpu}: {identity}: {lines[-1] if lines else "starting"}', flush=True)
                    last_status = time.monotonic()
                if active:
                    time.sleep(3)
        except BaseException:
            for process, handle, *_ in active.values():
                try:
                    stop_process(process)
                finally:
                    handle.close()
            raise
        return sorted(completed, key=lambda r: r['_identity'])


def train_jobs(configs, phase):
    return [(cfg['trial_id'], {'phase': phase, 'config': cfg}) for cfg in configs]


def run_training(pool, budget='extended'):
    configs = screen_plan(budget)
    print(f'Screen: {len(configs)} моделей, сумма эпох = {sum(c["epochs"] for c in configs)}')
    screening = pool.run('screen', train_jobs(configs, 'screen'))
    refinement = pool.run('refine', train_jobs(refinement_plan(screening, budget), 'screen'))
    finalists = pool.run('final', train_jobs(final_plan(screening + refinement, budget), 'final'))
    # Pick TWO completed configurations, then repeat each on predeclared seeds.
    leaders = sorted(finalists, key=lambda r: (-r['selection_metric'], r['trial_id']))[:2]
    seeds = (BASE_SEED + 101, BASE_SEED + 202) if budget == 'extended' else (BASE_SEED + 101,)
    repeats = []
    for leader in leaders:
        for seed in seeds:
            cfg = dict(leader['config'], seed=seed,
                       trial_id=leader['trial_id'] + f'_seed{seed}')
            repeats.append(cfg)
    confirmed = pool.run('confirm', train_jobs(repeats, 'final'))
    all_results = screening + refinement + finalists + confirmed
    rows = [{key: r.get(key) for key in ('trial_id', 'model', 'phase', 'selection_metric',
                                        'best_epoch', 'elapsed_seconds')}
            | {'seed': r['config']['seed'], 'config': json.dumps(r['config'], sort_keys=True)}
            for r in all_results]
    pd.DataFrame(rows).to_csv(pool.output_dir / 'experiments.csv', index=False)
    # Never choose the luckiest seed. All declared repeats enter their group.
    groups = []
    for leader in leaders:
        group = [leader] + [r for r in confirmed if r['trial_id'].startswith(leader['trial_id'] + '_seed')]
        groups.append(group)
    write_json(pool.output_dir / 'training_summary.json', {'finalists': finalists, 'groups': groups})
    return finalists, groups


def predict_jobs(records, split, tta):
    return [(r['trial_id'] + '_' + split + '_' + tta,
             {'phase': 'predict', 'config': r['config'], 'checkpoint': str(Path(r['_directory']) / 'best.pt'),
              'checkpoint_sha256': digest(Path(r['_directory']) / 'best.pt'),
              'split': split, 'tta': tta}) for r in records]


def aligned_mean(predictions):
    """Average probabilities with strict id/label/column alignment checks."""
    arrays = [np.load(p['_prefix'] + '.npz', allow_pickle=False) for p in predictions]
    try:
        ids = arrays[0]['ids'].astype(str)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError('Ожидаются уникальные одномерные id.')
        labels = arrays[0]['labels'].copy() if 'labels' in arrays[0].files else None
        class_ids = arrays[0]['class_ids']
        classes = arrays[0]['classes'].astype(str)
        shape = arrays[0]['probabilities'].shape
        if len(shape) != 2 or shape[0] != len(ids) or shape[1] != len(classes) or not np.array_equal(class_ids, np.arange(shape[1])):
            raise ValueError('Форма прогнозов не соответствует id или classes.')
        total = np.zeros(shape, dtype=np.float64)
        for values in arrays:
            if not np.array_equal(ids, values['ids'].astype(str)):
                raise ValueError('Порядок id отличается между прогнозами.')
            if ('labels' in values.files) != (labels is not None):
                raise ValueError('Смешаны val и test прогнозы.')
            if labels is not None and not np.array_equal(labels, values['labels']):
                raise ValueError('Метки отличаются между прогнозами.')
            if not np.array_equal(class_ids, values['class_ids']) or not np.array_equal(classes, values['classes'].astype(str)):
                raise ValueError('Порядок классов отличается между моделями.')
            prob = values['probabilities']
            if prob.shape != shape or not np.isfinite(prob).all():
                raise ValueError('Некорректные вероятности.')
            if (prob < 0).any() or not np.allclose(prob.sum(axis=1), 1, atol=1e-4):
                raise ValueError('Ожидаются нормированные softmax-вероятности.')
            total += prob
        return ids, (total / len(arrays)).astype(np.float32), labels
    finally:
        for values in arrays:
            values.close()


def select_recipe(pool, finalists, groups):
    rows, recipes, prediction_cache = [], [], {}
    # Include every full-train finalist as a single-model/no-TTA candidate.
    base_predictions = pool.run('predict_val', predict_jobs(finalists, 'val', 'none'))
    prediction_cache.update({p['_identity']: p for p in base_predictions})
    def add_recipe(name, members, tta):
        keys = [r['trial_id'] + '_val_' + tta for r in members]
        missing = [r for r, key in zip(members, keys) if key not in prediction_cache]
        if missing:
            new = pool.run('predict_val', predict_jobs(missing, 'val', tta))
            prediction_cache.update({p['_identity']: p for p in new})
        ids, probabilities, labels = aligned_mean([prediction_cache[k] for k in keys])
        correct = int((probabilities.argmax(1) == labels).sum())
        complexity = len(members) * {'none': 1, 'd4': 8, 'fivecrop': 5}[tta]
        row = dict(recipe=name, val_accuracy=correct / len(labels), correct=correct,
                   count=len(labels), forward_views=complexity)
        rows.append(row)
        recipes.append(dict(name=name, members=members, tta=tta, **row))
        return ids, probabilities, labels
    for r in finalists:
        add_recipe('single_' + r['trial_id'], [r], 'none')
    for i, group in enumerate(groups):
        add_recipe(f'seeds_group{i}', group, 'none')
    all_members = [r for group in groups for r in group]
    add_recipe('equal_ensemble', all_members, 'none')
    add_recipe('equal_ensemble_d4', all_members, 'd4')
    add_recipe('equal_ensemble_fivecrop', all_members, 'fivecrop')
    table = pd.DataFrame(rows).sort_values(['val_accuracy', 'forward_views'], ascending=[False, True])
    table.to_csv(pool.output_dir / 'validation_recipes.csv', index=False)
    winner = sorted(recipes, key=lambda r: (-r['val_accuracy'], r['forward_views']))[0]
    write_json(pool.output_dir / 'selected_recipe.json', winner)
    keys = [r['trial_id'] + '_val_' + winner['tta'] for r in winner['members']]
    ids, probabilities, labels = aligned_mean([prediction_cache[k] for k in keys])
    np.savez_compressed(pool.output_dir / 'selected_validation.npz', ids=ids,
                        probabilities=probabilities, labels=labels)
    print(table.to_string(index=False))
    print('Выбрано:', winner['name'], '| validation accuracy:', winner['val_accuracy'])
    if winner['val_accuracy'] <= .65:
        print('ВНИМАНИЕ: validation пока не превышает 65%. Улучшение не подтверждено.')
    return winner


def make_submission(pool, recipe, filename='submission_geo.csv'):
    # The recipe is already frozen. Test is used only for inference/export.
    predictions = pool.run('predict_test', predict_jobs(recipe['members'], 'test', recipe['tta']))
    ids, probabilities, _ = aligned_mean(predictions)
    test = pd.read_csv(Path(pool.data_dir) / 'test.csv', dtype={'id': str})
    if not np.array_equal(ids, test['id'].to_numpy(dtype=str)) or not test.id.is_unique:
        raise ValueError('Нарушен порядок или уникальность test id.')
    submission = pd.DataFrame({'id': ids, 'label': probabilities.argmax(1)})
    template_path = Path(pool.data_dir) / 'sample_submission_example.csv'
    if template_path.exists():
        sample = pd.read_csv(template_path, dtype={'id': str})
        if list(sample.columns) != ['id', 'label']:
            raise ValueError('Проверь формат sample_submission_example.csv.')
        if len(sample) == len(submission) and sample.id.is_unique and set(sample.id) == set(ids):
            submission = submission.set_index('id').loc[sample.id].reset_index()
        else:
            print('sample_submission_example — пример формата, не полный test; сохранён порядок test.csv.')
    if not submission.label.between(0, probabilities.shape[1] - 1).all():
        raise ValueError('Неверные индексы классов.')
    if Path(filename).name != filename or not filename.endswith('.csv'):
        raise ValueError('filename должен быть именем CSV без каталогов.')
    destination = pool.output_dir / filename
    submission.to_csv(destination, index=False)
    print('Готовый submission:', destination, '| строк:', len(submission))
    return destination


def run_pipeline(data_dir, output_dir, budget='extended', workers=2):
    pool = GPUPool(data_dir, output_dir, workers)
    finalists, groups = run_training(pool, budget)
    recipe = select_recipe(pool, finalists, groups)
    return make_submission(pool, recipe)
