import csv
import importlib.util
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HARNESS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_atlas_eval_waymo', HARNESS / 'waymo/run_atlas_eval.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def exact_iou(pred, gt):
    return np.asarray([[1.0 if np.allclose(p, gt[0]) else 0.0] for p in pred])


class AttackSummaryTests(unittest.TestCase):
    def test_injection_removal_and_class_matching(self):
        box = [0, 0, 0, 4, 2, 1.5, 0]
        infos = [dict(frame_id='a', spoof_gt=box + [1]), dict(frame_id='b', spoof_gt=box + [2]),
                 dict(frame_id='c', spoof_gt=None)]
        annos = [dict(frame_id=f, boxes_lidar=np.array([box]), name=np.array(['Vehicle'])) for f in 'abc']
        inject = run.attack_summary(infos, annos, 'injection', exact_iou)
        self.assertEqual((inject['num_targets'], inject['num_successes']), (2, 1))  # Vehicle box != Pedestrian target
        removal = run.attack_summary(infos, annos, 'removal', exact_iou)
        self.assertEqual(removal['num_successes'], 1)

    def test_misaligned_and_empty(self):
        infos = [dict(frame_id='a', spoof_gt=None)]
        with self.assertRaises(ValueError):
            run.attack_summary(infos, [dict(frame_id='b')], 'injection', exact_iou)
        with self.assertRaises(ValueError):
            run.attack_summary(infos, [dict(frame_id='a', boxes_lidar=np.zeros((0, 7)), name=np.array([]))],
                               'injection', exact_iou)

    def test_flatten_adds_class_means(self):
        metrics = {f'OBJECT_TYPE_TYPE_{t}_LEVEL_2/AP': v for t, v in (('VEHICLE', .6), ('PEDESTRIAN', .3),
                                                                      ('CYCLIST', .3))}
        flat = run.flatten(dict(model='m', dataset='d', metrics=dict(metrics, **{'recall/rcnn_0.3': .8})))
        self.assertAlmostEqual(flat['L2_mAP'], .4)
        self.assertEqual((flat['Vehicle_L2_AP'], flat['recall_rcnn_0.3']), (.6, .8))


class SchedulerTests(unittest.TestCase):
    def test_gpu_isolation_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for model in ('m1', 'm2'):
                (root / model / 'tools').mkdir(parents=True)
            runs = [(m, root / m, root / f'{m}.yaml', root / f'{m}.pth', 'python') for m in ('m1', 'm2')]
            args = SimpleNamespace(output_dir=root / 'out', data_root=root / 'data', datasets=['global_easy', 'removal_20'],
                                   gpus='0,1,2', exclude_gpus='2', max_parallel=None, slots_per_gpu=1,
                                   threads_per_task=1, batch_size=1, workers=0, save_to_file=False, resume=False)
            active, lock = set(), threading.Lock()

            def child(command, env, **kwargs):
                gpu = env['CUDA_VISIBLE_DEVICES']
                with lock:
                    self.assertNotIn(gpu, active)
                    active.add(gpu)
                self.assertNotEqual(gpu, '2')
                out = Path(command[command.index('--output_dir') + 1]) / command[command.index('--datasets') + 1]
                (out / 'attack_summary.json').write_text(json.dumps(dict(asr=.25, num_targets=4, num_successes=1)))
                (out / 'metrics.json').write_text(json.dumps({'recall/rcnn_0.3': .8}))
                time.sleep(.01)
                with lock:
                    active.remove(gpu)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(run.common, 'sha', return_value='sha'), \
                    patch.object(run.subprocess, 'run', side_effect=child) as fake:
                run.run_parallel(args, runs)
                self.assertEqual(fake.call_count, 4)
            rows = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual({(r['model'], r['dataset']) for r in rows},
                             {(m, n) for m in ('m1', 'm2') for n in args.datasets})
            with (root / 'out/summary.csv').open() as f:
                self.assertIn('recall_rcnn_0.3', next(csv.reader(f)))
            args.resume = True
            with patch.object(run.common, 'sha', return_value='sha'), \
                    patch.object(run.subprocess, 'run', side_effect=AssertionError('should skip')):
                run.run_parallel(args, runs)


if __name__ == '__main__':
    unittest.main()
