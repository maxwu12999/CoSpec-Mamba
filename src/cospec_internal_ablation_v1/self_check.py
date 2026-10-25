#!/usr/bin/env python3
"""CPU mathematical, control, data-identity, train/test-resume and aggregation checks.

The synthetic cache has no relationship to real classification accuracy.
"""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import bootstrap
import numpy as np
import torch
from common import write_json, read_json, file_hash, job_dir, recipe_id, SafeStop
from data import all_cases, mask_hash, validate_caches
from external_common import seed_all, state_hash
from suite_models import build_model as archived_build
from internal_models import build, VARIANTS
from run import prepare, run_one, model_for, code_identity
from worker import evaluate, Progress
from summarize import summarize

torch.set_num_threads(1)


def make_cache(root):
    rng = np.random.default_rng(77)
    for name in ('HongHu','PaviaU'):
        cache = root/'cache'/name; cache.mkdir(parents=True)
        cube = rng.normal(size=(5,6,17)).astype('float32')
        tr = np.zeros((5,6),np.int64); te=tr.copy()
        tr.flat[:8] = np.tile([1,2],4)
        te.flat[8:20] = np.tile([1,2],6)
        for fn, x in [('cube.npy',cube),('tr.npy',tr),('te.npy',te),('wave.npy',np.linspace(0,1,17,dtype='float32'))]:
            np.save(cache/fn,x)
        np.savez(cache/'stats.npz', means=np.zeros((2,17),np.float32), variances=np.ones((2,17),np.float32))
        full = np.ones(17,bool)
        cases=[dict(name='full', fraction=1., repeat=0,mask=full.tolist(),sha256=mask_hash(full))]
        g=np.random.default_rng(20261014)
        for rep in (1,2,3):
            order=g.permutation(17)
            for frac in (.8,.6,.4):
                mask=np.zeros(17,bool); mask[order[:round(17*frac)]]=True
                cases.append(dict(name=f'keep{round(frac*100)}_r{rep}', fraction=frac,repeat=rep,
                                  mask=mask.tolist(),sha256=mask_hash(mask)))
        write_json(cache/'data.json',dict(dataset=name,shape=list(cube.shape),classes=2,train_count=8,
            test_count=12,data_id='SYNTHETIC_'+name,cases=cases,normalization={'fit_on':'synthetic'},
            cache_sha256={n:file_hash(cache/n) for n in ('cube.npy','tr.npy','te.npy','stats.npz','wave.npy')}))


def configuration(reference, output, variants=('RDelta',), seeds=(1,)):
    args=SimpleNamespace(reference=reference,output=output,synthetic=True,device='cpu',
        svd_rtol=1e-6,datasets=['HongHu'],variants=list(variants),seeds=list(seeds))
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare(args)


class InternalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='cospec_internal_checks_')
        cls.root=Path(cls.temp.name); cls.reference=cls.root/'reference'
        make_cache(cls.reference)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def small_model(self, variant):
        seed_all(3)
        wave=np.linspace(0,1,17,dtype='float32')
        stats=dict(means=np.zeros((2,17),np.float32),variances=np.ones((2,17),np.float32))
        return build(variant,wave,stats,3,backend='reference',patch=3,dim=8,depth=4,basis=8)

    def test_01_archived_bytes_and_identical_initialization(self):
        self.assertEqual(len(code_identity()),64)
        models={v:self.small_model(v) for v in VARIANTS}
        self.assertEqual(len({state_hash(m.backbone_state()) for m in models.values()}),1)
        self.assertEqual(len({state_hash(models[v].operator.projection.state_dict()) for v in ('LS','R','RDelta','D')}),1)
        for v in ('LS','R'):
            self.assertFalse(any('refiner' in n for n,_ in models[v].named_parameters()))
        self.assertEqual(models['D'].component_counts()['total'],models['RDelta'].component_counts()['total'])
        self.assertEqual(models['D'].component_counts()['total']-models['R'].component_counts()['total'],8*8+8)

    def test_02_minimum_norm_svd_with_64_coefficients_and_rank_deficiency(self):
        wave=np.linspace(0,1,17,dtype='float32')
        stats=dict(means=np.zeros((2,17),np.float32),variances=np.ones((2,17),np.float32))
        seed_all(3)
        op=build('LS',wave,stats,3,backend='reference',patch=3,dim=8,depth=4,basis=64).operator
        x=torch.randn(4,4,17)
        mask=torch.ones(4,17,dtype=torch.bool);mask[1,3:]=False; mask[2,1:]=False;mask[3]=mask[1]
        contaminated=x.clone();contaminated[~mask[:,None,:].expand_as(x)]=float('nan')
        coeff=op.coefficients(contaminated,mask)
        phi=op.phi.numpy().astype('float64')
        for i in range(4):
            observed=mask[i].numpy()
            target=x[i].numpy()[:,observed] @ np.linalg.pinv(phi[observed],rcond=1e-6).T
            np.testing.assert_allclose(coeff[i].numpy(),target,rtol=3e-4,atol=3e-4)
        self.assertTrue(torch.isfinite(coeff).all())
        with self.assertRaises(ValueError):
            op.coefficients(x,torch.zeros_like(mask))

    def test_03_refiner_and_observation_constraint_equations(self):
        r=self.small_model('R').operator;rd=self.small_model('RDelta').operator;d=self.small_model('D').operator
        x=torch.randn(2,9,17);mask=torch.ones(2,17,dtype=torch.bool);mask[:,4:]=False
        clean=torch.where(mask[:,None,:],x,0.)
        torch.testing.assert_close(r(clean,mask)['features'],d(clean,mask)['features'],rtol=0,atol=0)
        with torch.no_grad():
            rd.refiner.bias.fill_(.7);d.refiner.bias.fill_(.7)
        c=r.coefficients(clean,mask)
        correction=.1*torch.tanh(rd.refiner(c))
        torch.testing.assert_close(rd(clean,mask)['coefficients'],c+correction)
        phi=d.phi;g=torch.einsum('ck,bc,cl->bkl',phi,mask.float(),phi)
        chol=torch.linalg.cholesky(g+.01*d.identity)
        resolution=torch.cholesky_solve(g,chol)
        target=c+torch.einsum('bkl,bsl->bsk',resolution,correction)
        torch.testing.assert_close(d(clean,mask)['coefficients'],target)
        self.assertFalse(torch.allclose(target,c+correction))
        loss=rd(clean,mask)['features'][...,0].sum();loss.backward()
        self.assertGreater(rd.refiner.weight.grad.abs().sum().item(),0)

    def test_04_D_is_exact_archived_model_and_hidden_values_are_ignored(self):
        d=self.small_model('D').eval()
        seed_all(3)
        old=archived_build('D',np.linspace(0,1,17,dtype='float32'),
            dict(means=np.zeros((2,17),np.float32),variances=np.ones((2,17),np.float32)),3,
            dict(patch=3,dim=8,depth=4,basis=8,backend='reference')).eval()
        self.assertEqual(state_hash(d.state_dict()),state_hash(old.state_dict()))
        x=torch.randn(2,17,3,3);mask=torch.ones(2,17,dtype=torch.bool);mask[:,7:]=False
        with torch.no_grad():
            torch.testing.assert_close(d(x,mask)['logits'],old(x,mask)['logits'],rtol=0,atol=0)
            bad=x.clone();bad[:,7:]=float('nan')
            for v in VARIANTS:
                m=self.small_model(v).eval()
                torch.testing.assert_close(m(x,mask)['logits'],m(bad,mask)['logits'],rtol=0,atol=0)

    def test_05_mask_nesting_and_corrupt_cache_rejection(self):
        meta=read_json(self.reference/'cache/HongHu/data.json');cases=all_cases(meta)
        self.assertEqual(len(cases),16)
        for repeat in (1,2,3):
            rows=[c for c in cases if c.get('repeat')==repeat]
            for a,b in zip(rows,rows[1:]):
                self.assertTrue(np.all(np.array(b['mask'])<=np.array(a['mask'])))
        p=self.reference/'cache/HongHu/wave.npy';saved=p.read_bytes()
        try:
            p.write_bytes(saved+b'corrupt')
            with self.assertRaises(ValueError):
                validate_caches(self.reference,self.root/'corrupt_out_SYNTHETIC',('HongHu',),synthetic=True)
        finally:
            p.write_bytes(saved)

    def test_06_training_resume_is_bit_identical(self):
        one=configuration(self.reference,self.root/'continuous_SYNTHETIC')
        two=configuration(self.reference,self.root/'resumed_SYNTHETIC')
        a,b=one['jobs'][0],two['jobs'][0]
        with contextlib.redirect_stdout(io.StringIO()):
            run_one(one,a,lambda:False)
            with self.assertRaises(SafeStop):
                run_one(two,b,lambda:False,stop_after_epoch=1)
            run_one(two,b,lambda:False)
        ra=read_json(job_dir(one,a)/'result.json');rb=read_json(job_dir(two,b)/'result.json')
        self.assertEqual(ra['model_sha256'],rb['model_sha256'])
        ha=read_json(job_dir(one,a)/'history.json');hb=read_json(job_dir(two,b)/'history.json')
        self.assertEqual([v['loss'] for v in ha.values()],[v['loss'] for v in hb.values()])

    def test_07_evaluation_resume_has_no_duplicate_test_pixels(self):
        cfg=configuration(self.reference,self.root/'eval_resume_SYNTHETIC',variants=('R',))
        job=cfg['jobs'][0];model=model_for(cfg,job).eval();sha=state_hash(model.state_dict())
        directory=job_dir(cfg,job);directory.mkdir(parents=True)
        calls=[0]
        def stop():
            calls[0]+=1
            return calls[0]>=2
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SafeStop):
                evaluate(model,cfg,job,directory,sha,Progress(directory/'progress.json',recipe_id(cfg,job)),stop)
            resumed=evaluate(model,cfg,job,directory,sha,Progress(directory/'progress.json',recipe_id(cfg,job)),lambda:False)
            other=directory.parent/'fresh';other.mkdir()
            fresh=evaluate(model,cfg,job,other,sha,Progress(other/'progress.json',recipe_id(cfg,job)),lambda:False)
        self.assertEqual(len(resumed),16)
        for a,b in zip(resumed,fresh):
            self.assertEqual(a['confusion_matrix'],b['confusion_matrix'])
            self.assertEqual(a['oa_percent'],b['oa_percent'])
            self.assertEqual(a['count'],12)

    def test_08_statistics_use_three_seed_means_and_paired_differences(self):
        cfg=configuration(self.reference,self.root/'summary_SYNTHETIC',variants=('LS','R'),seeds=(1,2,3))
        for job in cfg['jobs']:
            rows=[]
            for c in cfg['data']['HongHu']['cases']:
                value=job['seed']*10+(job['seed'] if job['model']=='R' else 0)
                value+=(-2,0,2)[c['repeat']-1] if c['fraction']<1 else 0
                rows.append(dict(case=c['name'],fraction=c['fraction'],repeat=c['repeat'],oa_percent=value,
                    aa_percent=value,kappa=.3,recipe_id=recipe_id(cfg,job),model_sha256='fixture',
                    mask_sha256=c['sha256'],count=12))
            write_json(job_dir(cfg,job)/'result.json',dict(complete=True,recipe_id=recipe_id(cfg,job),
                completed_epochs=2,model_sha256='fixture',metrics=rows))
        summarize(cfg['output'])
        folder=Path(cfg['output'])/'summary'
        rows=list(csv.DictReader((folder/'table6_oa.csv').open(encoding='utf-8-sig')))
        self.assertEqual(next(x for x in rows if x['variant'].startswith('LS '))['keep60'],'20.00 ± 10.00')
        self.assertEqual(next(x for x in rows if x['variant'].startswith('B '))['keep60'],'TBD')
        pairs=list(csv.DictReader((folder/'paired_differences.csv').open(encoding='utf-8-sig')))
        self.assertEqual(next(x for x in pairs if x['contrast']=='R-LS' and x['fraction']=='0.6')['delta_OA_pp'],'2.00 ± 1.00')

    def test_09_production_15_by_15_64_basis_four_layers_all_variants(self):
        wave=np.linspace(0,1,103,dtype='float32')
        stats=dict(means=np.zeros((9,103),np.float32),variances=np.ones((9,103),np.float32))
        x=torch.randn(1,103,15,15);mask=torch.ones(1,103,dtype=torch.bool);mask[:,41:]=False
        common=[]
        for variant in VARIANTS:
            seed_all(1)
            model=build(variant,wave,stats,1,backend='reference').eval()
            self.assertEqual(len(model.layers),4)
            self.assertEqual(model.position_encoder.out_features,64)
            common.append(state_hash(model.backbone_state()))
            with torch.no_grad():
                logits=model(x,mask)['logits']
            self.assertEqual(tuple(logits.shape),(1,9))
            self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(len(set(common)),1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InternalChecks))
    if args.report:
        write_json(args.report,dict(scope='synthetic_implementation_checks_only',torch=str(torch.__version__),
            numpy=np.__version__,device='cpu',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            passed=result.wasSuccessful(),formal_accuracy_experiments_run=False,cuda_kernel_tested=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)

