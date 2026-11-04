import csv
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HARNESS = Path(__file__).resolve().parents[1]
REPO = HARNESS.parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = load(HARNESS / 'nuscenes/run_atlas_eval.py', 'run_atlas_eval_nusc')
atlas = load(HARNESS / 'nuscenes/atlas_eval.py', 'atlas_eval_test')
metrics = load(HARNESS / 'nuscenes/attacked_metrics.py', 'attacked_metrics_test')

try:
    import shapely  # noqa: F401
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False


def box_iou(boxes, target):
    """Axis-aligned stand-in for the rotated IoU: enough for matching logic."""
    return np.asarray([1.0 if np.allclose(b[:7], target[:7]) else 0.0 for b in boxes])


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class GeometryTests(unittest.TestCase):
    def test_det3d_box_conversion(self):
        out = dict(box3d_lidar=Tensor([[1, 2, 3, 2, 6, 4, .3, .7, -np.pi / 2]]), scores=Tensor([.9]),
                   label_preds=Tensor([0]), metadata={'token': 'a'})
        anno = run.normalize_det3d(out, ['car'])
        np.testing.assert_allclose(anno['boxes_lidar'], [[1, 2, 3, 6, 2, 4, 0, .3, .7]])
        self.assertEqual((list(anno['name']), list(anno['pred_labels'])), (['car'], [1]))

    @unittest.skipUnless(HAVE_SHAPELY, 'shapely not installed')
    def test_rotated_iou_and_height(self):
        a = np.array([0, 0, 0, 6, 2, 2, 0.])
        b = a.copy()
        b[6] = np.pi / 2
        c = a.copy()
        c[2] = 3
        np.testing.assert_allclose(run.oriented_iou([a, b, c], a), [1, .2, 0], atol=1e-7)

    def test_attacked_points_replace_sweeps(self):
        with tempfile.TemporaryDirectory() as d:
            points = np.array([[1, 2, 3, 42, .1, 9], [4, 5, 6, 55, .2, 9]], dtype=np.float32)
            np.save(Path(d) / 'a.npy', points)
            res, _ = run.AtlasPoints(d, {'a': {'points_file': 'a.npy'}})({'lidar': {}}, {'token': 'a', 'sweeps': ['x']})
        np.testing.assert_allclose(res['lidar']['combined'], points[:, :5])
        np.testing.assert_allclose(res['lidar']['times'], points[:, 4:5])


class AsrTests(unittest.TestCase):
    def setUp(self):
        self.target = [0, 0, 0, 6, 2, 2, 0]
        self.dataset = SimpleNamespace(atlas_by_token={'a': dict(spoof_gt=self.target, segment='s', frame_index=0,
                                                                 atk_n_spoof_k=0)})

    def score(self, attack, config, names, cls=None, iou=.3):
        annos = [dict(metadata={'token': 'a'}, boxes_lidar=np.array([self.target] * len(names)),
                      score=np.full(len(names), .9), name=np.array(names))]
        args = SimpleNamespace(atlas_attack=attack, atlas_config=config, atlas_target_class=cls,
                               atlas_iou_thresh=iou, atlas_score_thresh=0)
        with tempfile.TemporaryDirectory() as d:
            result = atlas.evaluate(self.dataset, annos, args, Path(d), logging.getLogger('test'), iou_fn=box_iou)
            summary = json.loads((Path(d) / 'atlas_asr.json').read_text())['summary']
        return result['atlas/asr'], summary

    def test_injection_and_removal(self):
        self.assertEqual(self.score('inject', 'inject_global_easy', ['car'])[0], 1.)
        self.assertEqual(self.score('removal', 'removal_az20', ['car'])[0], 0.)

    def test_target_class_from_configuration(self):
        self.assertEqual(atlas.target_class_for('inject_relative_pedestrian_hard'), 'pedestrian')
        self.assertEqual(atlas.target_class_for('inject_global_cyclist_easy'), 'bicycle')
        self.assertEqual(atlas.target_class_for('removal_az10'), 'car')
        self.assertEqual(run.target_class_for('inject_global_cyclist_easy'), 'bicycle')
        asr, summary = self.score('inject', 'inject_global_pedestrian_easy', ['car'])
        self.assertEqual((asr, summary['target_class']), (0., 'pedestrian'))
        asr, summary = self.score('inject', 'inject_global_pedestrian_easy', ['pedestrian'])
        self.assertEqual((asr, summary['num_target_fully_occluded']), (1., 1))

    def test_missing_predictions_refused(self):
        args = SimpleNamespace(atlas_attack='inject', atlas_config='inject_global_easy', atlas_target_class=None,
                               atlas_iou_thresh=.3, atlas_score_thresh=0)
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            atlas.evaluate(self.dataset, [], args, Path(d), logging.getLogger('test'), iou_fn=box_iou)

    def test_default_iou_per_class(self):
        import argparse
        p = argparse.ArgumentParser()
        atlas.add_arguments(p)
        for config, iou in (('inject_global_easy', .3), ('inject_global_pedestrian_easy', .5)):
            args = p.parse_args(['--atlas_attack', 'inject', '--atlas_data', '/d', '--atlas_config', config])
            atlas.validate_arguments(args, p)
            self.assertEqual(args.atlas_iou_thresh, iou)


class Boxes:
    def __init__(self):
        self.boxes = {}

    @property
    def sample_tokens(self):
        return list(self.boxes)

    def __getitem__(self, k):
        return self.boxes[k]

    def add_boxes(self, k, items):
        self.boxes[k] = items


class AttackedMetricTests(unittest.TestCase):
    def test_subset_keeps_empty_samples_and_copies(self):
        boxes = Boxes()
        boxes.add_boxes('attacked', [{'x': 1}])
        boxes.add_boxes('empty', [])
        boxes.add_boxes('clean', [{'x': 2}])
        selected = metrics.subset_boxes(boxes, {'attacked', 'empty'})
        self.assertEqual(set(selected.sample_tokens), {'attacked', 'empty'})
        self.assertEqual(selected['empty'], [])
        selected['attacked'][0]['x'] = 99
        self.assertEqual(boxes['attacked'][0]['x'], 1)
        for bad in ({'outside'}, set()):
            with self.assertRaises(ValueError):
                metrics.subset_boxes(boxes, bad)


class SchedulerTests(unittest.TestCase):
    def fixture(self, root):
        data = root / 'nusc'
        (data / 'v1.0-trainval').mkdir(parents=True)
        (data / 'v1.0-trainval/sample.json').write_text('[]')
        (root / 'pcdet/pcdet/datasets/nuscenes').mkdir(parents=True)
        (root / 'pcdet/pcdet/datasets/nuscenes/nuscenes_utils.py').touch()
        models = {}
        for name, backend in (('a', 'det3d'), ('b', 'pcdet')):
            (root / name).mkdir()
            (root / name / 'cfg').touch()
            (root / name / 'ep1.pth').touch()
            (root / name / 'ep2.pth').touch()
            models[name] = dict(backend=backend, repo=name, cfg='cfg', ckpt=f'{name}/ep{{epoch}}.pth', epochs=[1, 2])
        (root / 'models.json').write_text(json.dumps(models))
        configs = ['inject_global_easy', 'inject_global_pedestrian_easy', 'removal_az20']
        for config in configs:
            (root / 'atlas/data' / config).mkdir(parents=True)
            (root / 'atlas/data' / config / 'train-00000-of-00001.parquet').touch()
        return ['--manifest', str(root / 'models.json'), '--atlas_data', str(root / 'atlas'),
                '--dataroot', str(data), '--output_dir', str(root / 'out'), '--pcdet_repo', str(root / 'pcdet')]

    def test_jobs_gpu_isolation_slots_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            argv = self.fixture(root) + ['--gpus', '0,1', '--slots_per_gpu', '2', '--configs',
                                         'inject_global_easy', 'inject_global_pedestrian_easy', 'removal_az20',
                                         run.CLEAN, '--resume']
            active, seen, lock, calls = {}, set(), threading.Lock(), []

            def child(command, env, **kwargs):
                gpu = env['CUDA_VISIBLE_DEVICES']
                with lock:
                    active[gpu] = active.get(gpu, 0) + 1
                    self.assertLessEqual(active[gpu], 2)
                    seen.add(gpu)
                    calls.append(command)
                self.assertNotIn('WORLD_SIZE', env)
                out = Path(command[command.index('--output_dir') + 1])
                (out / 'values.json').write_text(json.dumps(dict(asr=.5, mAP=.3, NDS=.4)))
                time.sleep(.005)
                with lock:
                    active[gpu] -= 1
                return subprocess.CompletedProcess(command, 0)

            with patch.object(run.common, 'sha', return_value='sha'), \
                    patch.object(run.subprocess, 'run', side_effect=child) as fake:
                run.main(argv)
                self.assertEqual(fake.call_count, 16)  # 4 models (2 x 2 epochs) x 4 configurations
                run.main(argv)
                self.assertEqual(fake.call_count, 16)  # everything skipped on resume
            self.assertEqual(seen, {'0', '1'})
            by_config = {c[c.index('--configs') + 1]: c for c in calls}
            ped = by_config['inject_global_pedestrian_easy']
            self.assertEqual(ped[ped.index('--target_class') + 1], 'pedestrian')
            self.assertEqual(ped[ped.index('--iou_thresh') + 1], '0.5')
            self.assertNotIn('--atlas_data', by_config[run.CLEAN])
            rows = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(r['status'] == 'ok' for r in rows))
            self.assertEqual({r['epoch'] for r in rows}, {1, 2})
            with (root / 'out/summary.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 16)

    def test_changed_inputs_refuse_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            argv = self.fixture(root) + ['--gpus', '0', '--models', 'a_ep1', '--configs', 'removal_az20', '--resume']

            def child(command, env, **kwargs):
                out = Path(command[command.index('--output_dir') + 1])
                (out / 'values.json').write_text('{"asr": 1.0}')
                return subprocess.CompletedProcess(command, 0)
            with patch.object(run.subprocess, 'run', side_effect=child):
                with patch.object(run.common, 'sha', return_value='v1'):
                    run.main(argv)
                with patch.object(run.common, 'sha', return_value='v2'), self.assertRaises(SystemExit):
                    run.main(argv)
            row = json.loads((root / 'out/summary.json').read_text())[0]
            self.assertEqual(row['status'], 'ok')  # the failed retry did not overwrite the success

    def test_dry_run_discovers_configurations_without_gpus(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            argv = self.fixture(root) + ['--dry_run']
            with patch.dict('os.environ', {}, clear=False) as env:
                env.pop('CUDA_VISIBLE_DEVICES', None)
                run.main(argv)
            self.assertFalse((root / 'out').exists())


class PatchConsistencyTests(unittest.TestCase):
    def test_detector_patches_ship_the_harness_hook(self):
        """The hook installed into detector repositories is byte-identical to the harness copy."""
        hook = (HARNESS / 'nuscenes/atlas_eval.py').read_text().splitlines()
        patches = sorted((REPO / 'eval/detector-patches').glob('*/atlas-nuscenes.patch'))
        self.assertTrue(patches)
        for path in patches:
            lines = path.read_text().splitlines()
            start = lines.index('+++ b/pcdet/datasets/nuscenes/atlas_eval.py') + 2
            body = []
            for line in lines[start:]:
                if not line.startswith('+'):
                    break
                body.append(line[1:])
            self.assertEqual(body, hook, path)


if __name__ == '__main__':
    unittest.main()
