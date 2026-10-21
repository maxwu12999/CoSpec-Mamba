"""Aligned seed1 features and test-center maps for all six retention levels."""
import colorsys
from pathlib import Path
import numpy as np
import torch
from common import read_json, write_json, task_key, recipe_id, SafeStop
from data import Patches, feature_coordinates


def palette(classes):
    p = np.zeros((classes + 1, 3), np.uint8)
    for i in range(1, classes + 1):
        p[i] = np.array(colorsys.hsv_to_rgb((i * .61803398875) % 1, .75, .95)) * 255
    return p


def export_all(model, config, job, directory, model_sha, progress, stop):
    artifact = directory / 'artifacts.json'
    if job['seed'] != 1:
        write_json(artifact, dict(complete=True, status='not_requested', reason='seed1 maps/features; metrics use seeds1,2,3'))
        return
    meta = config['data'][job['dataset']]
    chosen = [c for c in meta['cases'] if c['name'] == 'full' or c['repeat'] == 1]
    coords = feature_coordinates(meta)
    dataset = Patches(meta, 'te', job['patch'], coordinates=coords)
    root = Path(config['output'])
    prefix = Path(task_key(job))
    features_root, maps_root = root / 'features' / prefix, root / 'maps' / prefix
    features_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i, case in enumerate(chosen):
        target = features_root / (case['name'] + '.npz')
        marker = target.with_suffix('.json')
        wanted = dict(recipe_id=recipe_id(config, job), model_sha256=model_sha,
                      mask_sha256=case['sha256'], count=len(dataset))
        if target.exists() and marker.exists() and read_json(marker) == wanted:
            continue
        if stop():
            raise SafeStop()
        embeddings, labels = [], []
        mask = torch.tensor(case['mask'], dtype=torch.bool, device=config['device'])
        with torch.inference_mode():
            for start in range(0, len(dataset), config['train']['eval_batch']):
                x, y = dataset.batch(range(start, min(len(dataset), start + config['train']['eval_batch'])))
                result = model(x.to(config['device']), mask[None].expand(len(y), -1))
                emb = result['embedding'].detach().cpu().numpy()
                if len(emb) != len(y) or not np.isfinite(emb).all():
                    raise ValueError('导出特征包含异常值')
                embeddings.append(emb)
                labels.append(y.numpy())
                progress.update(phase='features', export_done=i/len(chosen), case=case['name'])
                if stop():
                    raise SafeStop()
        from worker import atomic_npz
        atomic_npz(target, features=np.concatenate(embeddings), labels=np.concatenate(labels),
                   positions=coords, observed=np.asarray(case['mask'], bool))
        write_json(marker, wanted)
    # Release model activations/allocator cache before CPU-only map compression.
    if config['device'].startswith('cuda'):
        torch.cuda.empty_cache()
    from PIL import Image
    for i, case in enumerate(chosen):
        if stop():
            raise SafeStop()
        progress.update(force=True, phase='maps', export_done=(i+1)/len(chosen))
        target = maps_root / case['name']
        target.mkdir(parents=True, exist_ok=True)
        with np.load(directory / (case['name'] + '.npz')) as npz:
            co, pred, truth = npz['positions'], npz['predictions'], npz['labels']
            ids = np.zeros(meta['shape'][:2], np.uint16)
            gt = np.zeros_like(ids)
            errors = np.zeros(meta['shape'][:2], np.uint8)
            ids[tuple(co.T)], gt[tuple(co.T)] = pred + 1, truth + 1
            errors[tuple(co.T)] = np.where(pred == truth, 1, 2)
            colors = palette(meta['classes'])
            for name, array in [('prediction_ids', ids), ('prediction_test', colors[ids]),
                                ('truth_test', colors[gt]), ('errors_test', np.array([[0,0,0],[45,160,65],[230,40,40]],np.uint8)[errors])]:
                temp = target / (name + '.tmp')
                Image.fromarray(array).save(temp, format='PNG')
                temp.replace(target / (name + '.png'))
            write_json(target/'map_scope.json', dict(test_pixels=len(truth), unpredicted_value=0,
                       scope='original TE centers only, not an all-pixels scene prediction',
                       model_sha256=model_sha, mask_sha256=case['sha256']))
    write_json(artifact, dict(complete=True, status='completed', cases=[c['name'] for c in chosen],
               feature_sampling='fixed same class-balanced TE centers; at most50/class; not training inputs',
               tsne='raw embeddings exported; visualizations can be generated on CPU later',
               speed_scope='concurrent runtime is not a fair isolated GPU speed comparison'))
