import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


class GpuTests(unittest.TestCase):
    def test_slots_follow_cuda_visible_devices(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,5'}):
            self.assertEqual(common.gpu_slots(), ['3', '5'])

    def test_exclusion_limit_and_slots_per_gpu(self):
        self.assertEqual(common.gpu_slots('0,7,3', exclude='7'), ['0', '3'])
        self.assertEqual(common.gpu_slots('0,1,2', max_parallel=2, slots_per_gpu=2), ['0', '1', '0', '1'])
        with self.assertRaises(ValueError):
            common.gpu_slots('0,0')
        with self.assertRaises(ValueError):
            common.gpu_slots('7', exclude='7')
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            common.gpu_slots()

    def test_quarantine_removes_every_slot_of_a_gpu(self):
        pool = common.HealthyGpuPool(['7', '2', '7'])
        self.assertEqual(pool.acquire(), '7')
        pool.release('7', healthy=False)
        self.assertEqual(pool.acquire(), '2')
        pool.release('2', healthy=False)
        with self.assertRaisesRegex(RuntimeError, 'No healthy'):
            pool.acquire()

    def test_worker_env_is_single_gpu_and_not_distributed(self):
        with tempfile.TemporaryDirectory() as d:
            env = common.worker_env({'WORLD_SIZE': '8', 'RANK': '3', 'PYTHONPATH': 'inherited'}, '4', 2,
                                    [d, '/does/not/exist'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4')
        self.assertNotIn('WORLD_SIZE', env)
        self.assertNotIn('RANK', env)
        self.assertEqual(env['PYTHONPATH'], os.pathsep.join([d, 'inherited']))


class SummaryTests(unittest.TestCase):
    def test_failed_retry_keeps_success_and_models_merge(self):
        with tempfile.TemporaryDirectory() as d:
            ok = dict(model='a', config='c', status='ok', asr=.5)
            other = dict(model='b', config='c', status='ok', asr=.1)
            common.merge_summary(d, [ok], ('model', 'config'))
            common.merge_summary(d, [other], ('model', 'config'))
            common.merge_summary(d, [dict(ok, status='failed', asr=None)], ('model', 'config'))
            self.assertEqual(json.loads((Path(d) / 'summary.json').read_text()), [ok, other])
            self.assertEqual(len((Path(d) / 'summary.csv').read_text().splitlines()), 3)


class ManifestTests(unittest.TestCase):
    def write(self, d, data):
        path = Path(d) / 'sub' / 'models.json'
        path.parent.mkdir()
        path.write_text(json.dumps(data))
        return path

    def test_paths_env_and_epochs(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'CKPTS': '/ckpts'}):
            path = self.write(d, {
                '_comment': 'ignored',
                'm': dict(backend='pcdet', repo='../repo', cfg='tools/cfgs/x.yaml',
                          ckpt='${CKPTS}/m/epoch_{epoch}.pth', epochs=[9, 18], pythonpath=['deps']),
                'n': dict(backend='det3d', repo='/abs/repo', cfg='/abs/cfg.py', ckpt='n.pth'),
            })
            specs = common.load_manifest(path)
            base = path.parent
        self.assertEqual(sorted(specs), ['m_ep18', 'm_ep9', 'n'])
        m = specs['m_ep9']
        self.assertEqual(m['repo'], str((base / '../repo').resolve()))
        self.assertEqual(m['cfg'], str(Path(m['repo']) / 'tools/cfgs/x.yaml'))
        self.assertEqual(m['ckpt'], '/ckpts/m/epoch_9.pth')
        self.assertEqual((m['base_model'], m['epoch']), ('m', 9))
        self.assertEqual(m['pythonpath'], [str(base / 'deps')])
        self.assertEqual(specs['n']['ckpt'], str(base / 'n.pth'))
        self.assertNotIn('epoch', specs['n'])

    def test_default_repo(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, {'m': dict(cfg='tools/cfgs/x.yaml', ckpt='m.pth')})
            spec = common.load_manifest(path, default_repo='/repo')['m']
        self.assertEqual((spec['repo'], spec['cfg']), ('/repo', '/repo/tools/cfgs/x.yaml'))

    def test_invalid_manifests(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ({'m': dict(cfg='c')},
                        {'m': dict(cfg='c', ckpt='e_{epoch}.pth')},
                        {'m': dict(cfg='c', ckpt='e.pth', epochs=[1])},
                        {'../m': dict(cfg='c', ckpt='e.pth')},
                        {}):
                path = Path(d) / 'm.json'
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError, msg=bad):
                    common.load_manifest(path)


if __name__ == '__main__':
    unittest.main()
