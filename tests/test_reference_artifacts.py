"""Round-trip original split bytes and reject corrupted or incompatible artifacts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reference_artifacts', ROOT/'tools/reference_artifacts.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ReferenceArtifactsChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root/'reference/cache/Fixture'
        self.cache.mkdir(parents=True)
        raw = np.random.default_rng(5).normal(size=(5,6,7)).astype(np.float32)
        tr = np.zeros((5,6),np.int64)
        tr.flat[:6] = [1,2,1,2,1,2]
        te = np.zeros_like(tr)
        te.flat[6:16] = [1,2]*5
        training = raw[tr>0].astype(np.float64)
        mean, scale = float(training.mean()), float(training.std())
        normalized = ((raw-mean)/scale).astype(np.float32)
        for name, value in [('cube.npy',normalized),('tr.npy',tr),('te.npy',te),
                            ('wave.npy',np.linspace(0,1,7,dtype=np.float32))]:
            np.save(self.cache/name,value)
        np.savez(self.cache/'stats.npz',means=np.zeros((2,7),np.float32),variances=np.ones((2,7),np.float32))
        self.raw = self.root/'raw.npy'
        np.save(self.raw,raw)
        self.meta = dict(data_id='synthetic-test-only',shape=[5,6,7],train_count=6,test_count=10,
                         normalization=dict(global_mean=mean,global_scale=scale,fit_on='TR center spectra only'),
                         cache_sha256={n:mod.file_hash(self.cache/n) for n in mod.CACHE_FILES})
        (self.cache/'data.json').write_text(json.dumps(self.meta))
        self.identities = {'Fixture':self.meta}

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_preserves_exact_split_statistics_and_normalized_cube(self):
        out = self.root/'artifacts'
        mod.export_reference(self.root/'reference',out,['Fixture'],self.identities)
        self.assertFalse((out/'Fixture/cube.npy').exists())
        restored = mod.restore_reference(out,self.raw,self.root/'restored','Fixture',identities=self.identities)
        for name in (*mod.CACHE_FILES,'data.json'):
            self.assertEqual((self.cache/name).read_bytes(),(restored/name).read_bytes())

    def test_corruption_and_incompatible_raw_cube_are_rejected_without_completed_output(self):
        out = self.root/'artifacts'
        mod.export_reference(self.root/'reference',out,['Fixture'],self.identities)
        wrong = self.root/'wrong.npy'
        np.save(wrong,np.zeros((5,6,7),np.float32))
        with self.assertRaisesRegex(ValueError,'cube.npy'):
            mod.restore_reference(out,wrong,self.root/'restore-wrong','Fixture',identities=self.identities)
        self.assertFalse((self.root/'restore-wrong/cache/Fixture').exists())
        (out/'Fixture/tr.npy').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'corrupted'):
            mod.restore_reference(out,self.raw,self.root/'restore-corrupt','Fixture',identities=self.identities)

    def test_wrong_paper_identity_is_rejected_before_export(self):
        bad = dict(self.meta,data_id='different split')
        with self.assertRaisesRegex(ValueError,'identity'):
            mod.export_reference(self.root/'reference',self.root/'no-export',['Fixture'],{'Fixture':bad})
        self.assertFalse((self.root/'no-export').exists())


if __name__ == '__main__':
    unittest.main()
