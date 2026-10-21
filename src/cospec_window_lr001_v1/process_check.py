#!/usr/bin/env python3
"""Synthetic subprocess scheduler checks plus read-only parent archive checks."""
import argparse
import copy
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import zipfile

from common import (ROOT, file_hash, job_dir, make_jobs, profile_key, read_json,
                    task_key, write_json)
from window_protocol import (parent_jobs, prepare_config, snapshot_parent,
                             validate_parent, verify_science)
import scheduler


def materialize_parent(archive, root):
    with zipfile.ZipFile(archive) as reader:
        prefixes = [n for n in reader.namelist() if n.endswith('/config.json') and n.count('/') == 1]
        if len(prefixes) != 1:
            raise ValueError('归档根config.json缺失或重复')
        config = read_json_bytes(reader.read(prefixes[0]))
        prefix = prefixes[0].removesuffix('config.json')
        write_json(root / 'config.json', config)
        for job in parent_jobs(config):
            for name in ('result.json', 'history.json'):
                target = root / 'runs' / task_key(job) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(reader.read(prefix + 'runs/' + task_key(job) + '/' + name))
    return root


def read_json_bytes(value):
    import json
    return json.loads(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-output', type=Path, default=ROOT.parent / 'diagnostics/COSPEC_FULL_LR001_V1')
    parser.add_argument('--parent-archive', type=Path)
    args = parser.parse_args()
    manifest = verify_science()
    with tempfile.TemporaryDirectory(prefix='cospec_window_process_check_') as temporary:
        root = Path(temporary)
        parent_root = materialize_parent(args.parent_archive, root / 'parent') if args.parent_archive else args.parent_output
        before = {str(p): file_hash(p) for p in parent_root.rglob('*.json')}
        parent, records = validate_parent(parent_root, manifest)
        assert len(records) == 12
        hardware = dict(name='SIMULATED_NO_GPU', uuid='TEST_ONLY', total_mib=24564., driver='test')
        arguments = argparse.Namespace(output=root / 'prepared', reference=root / 'reference',
            parent_output=parent_root, max_parallel=10, eval_batch=128, headroom_mib=1024.)
        data = copy.deepcopy(parent['data'])
        data = {'HongHu': data['HongHu']}
        # The archive contains identity records, not the >1GB cube: only this
        # filesystem dependency is replaced here; real starts hash the real cube.
        with patch('data.validate_caches', return_value=data):
            config = prepare_config(arguments, hardware)
            again = prepare_config(arguments, hardware)
        assert config['protocol_id'] == again['protocol_id']
        assert len(config['jobs']) == 12 and config['train'] == parent['train']
        assert config['data_identity']['HongHu'] == parent['data_identity']['HongHu']
        audit = read_json(arguments.output / 'protocol_audit.json')
        assert audit['parent_epochs_checked'] == 1200
        # Rebinding paths must not mutate any original JSON or reuse old weights.
        assert before == {str(p): file_hash(p) for p in parent_root.rglob('*.json')}
        for target in (parent_root, parent_root / 'nested', root, ROOT):
            forbidden = copy.copy(arguments)
            forbidden.output = target
            from window_protocol import protect_output
            try:
                protect_output(target, arguments.reference, parent_root)
            except ValueError:
                pass
            else:
                raise AssertionError('未阻止覆盖旧目录')
        incompatible = copy.copy(arguments)
        incompatible.eval_batch = 256
        try:
            prepare_config(incompatible, hardware)
        except ValueError:
            pass
        else:
            raise AssertionError('允许改变测试batch')
        print('[通过] 真实9/15归档12项/1200轮/16掩码核对；新配置可重复生成；原JSON未修改。', flush=True)
        # Verify that a stage-LR history is rejected, using a disposable copy.
        bad_root = root / 'bad_parent'
        import shutil
        shutil.copytree(parent_root, bad_root)
        bad_config = dict(parent, output=str(bad_root))
        history_path = job_dir(bad_config, parent_jobs(parent)[0]) / 'history.json'
        history = read_json(history_path)
        history['73']['lr'] = .0001
        write_json(history_path, history)
        try:
            validate_parent(bad_root, manifest)
        except ValueError as error:
            assert '分段学习率' in str(error)
        else:
            raise AssertionError('允许把分段学习率结果标成固定0.001')
        print('[通过] 会拒绝分段学习率记录、不同batch和覆盖旧实验的输出目录。', flush=True)

        from self_check import fixture, TinyNet
        from task_worker import install
        import torch
        import worker
        import speed_data
        from results import complete
        torch.set_num_threads(1)
        tiny = fixture(root / 'end_to_end')
        tiny['train']['epochs'] = 2
        tiny['jobs'] = [make_jobs()[0]]
        install()
        worker.run(tiny, tiny['jobs'][0], lambda: False, factory=lambda *_: TinyNet())
        assert complete(tiny, tiny['jobs'][0])
        saved = job_dir(tiny, tiny['jobs'][0]) / 'last.pt'
        checksum = file_hash(saved)
        worker.run(tiny, tiny['jobs'][0], lambda: False, factory=lambda *_: TinyNet())
        assert file_hash(saved) == checksum
        artifacts = read_json(job_dir(tiny, tiny['jobs'][0]) / 'artifacts.json')
        assert artifacts['complete'] and len(artifacts['cases']) == 6
        speed_data.uninstall()
        print('[通过] 合成数据实际训练→16条件测试→6组分类图/特征→完成标记；重启不重复训练。', flush=True)

        cfg = fixture(root / 'scheduling')
        cfg['train']['epochs'] = 2
        cfg['jobs'] = make_jobs()
        output = Path(cfg['output'])
        output.mkdir()
        for folder in ('logs', 'profiles'):
            (output / folder).mkdir()
        snapshot_parent(parent, records, output)
        write_json(output / 'config.json', cfg)
        write_json(output / 'runtime_control.json', dict(max_parallel=10))
        for job in cfg['jobs']:
            key = profile_key(cfg, job)
            write_json(output / 'profiles' / (key + '.json'), dict(complete=True, profile_key=key,
                train_peak_reserved_mib=64., eval_peak_reserved_mib=64., train_batch_seconds=.01,
                eval_batch_seconds=.002, rss_peak_mib=64.))
        wrapper = root / 'wrapper'
        wrapper.mkdir()
        # No CUDA and no fabricated scientific results: these subprocesses only
        # emit small, explicitly synthetic records to exercise queue mechanics.
        (wrapper / 'task_worker.py').write_text('''import argparse,fcntl,sys,time
from pathlib import Path
sys.path.insert(0, CORE_PATH)
from common import read_json,write_json,job_dir,recipe_id,identity,batch_count
p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--index',type=int)
a=p.parse_args();c=read_json(a.config);j=c['jobs'][a.index];d=job_dir(c,j);d.mkdir(parents=True,exist_ok=True)
with (d/'run.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 t=c['data'][j['dataset']];train=batch_count(t['train_count'],c['train']['batch'])*2
 steps=((t['test_count']+c['train']['eval_batch']-1)//c['train']['eval_batch'])*16
 write_json(d/'progress.json',dict(recipe_id=recipe_id(c,j),phase='train',updated_unix=time.time(),
            train_steps_done=0,train_steps_total=train,test_steps_done=0,test_steps_total=steps,**identity()))
 marker=d/'injected_oom'
 if a.index==0 and not marker.exists():
  marker.write_text('SIMULATED OOM FOR QUEUE TEST ONLY');time.sleep(1.5)
  write_json(d/'outcome.json',dict(state='oom',error='SIMULATED queue OOM',recipe_id=recipe_id(c,j)))
  raise SystemExit(75)
 time.sleep(3)
 metrics=[dict(case=x['name'],fraction=x['fraction'],repeat=x['repeat'],mask_sha256=x['sha256'],
          count=t['test_count'],class_support=t['class_support'],oa_percent=33.,aa_percent=33.,kappa=0.) for x in t['cases']]
 write_json(d/'result.json',dict(complete=True,scope='SYNTHETIC_QUEUE_TEST_ONLY',
            recipe_id=recipe_id(c,j),job=j,completed_epochs=2,lr_policy='constant_0.001',metrics=metrics))
 write_json(d/'progress.json',dict(recipe_id=recipe_id(c,j),phase='completed',updated_unix=time.time(),
            train_steps_done=train,train_steps_total=train,test_steps_done=steps,test_steps_total=steps,**identity()))
'''.replace('CORE_PATH', repr(str(ROOT))), encoding='utf-8')
        old_root, old_ramp, old_poll = scheduler.ROOT, scheduler.RAMP_SECONDS, scheduler.POLL_SECONDS
        scheduler.ROOT, scheduler.RAMP_SECONDS, scheduler.POLL_SECONDS = wrapper, 0., .02
        actual_gpu, actual_processes, actual_ram = scheduler.gpu_info, scheduler.gpu_processes, scheduler.available_ram
        scheduler.gpu_info = lambda _: dict(hardware, free_mib=24564.)
        scheduler.gpu_processes = lambda _: {}
        scheduler.available_ram = lambda: 102400.
        deadline = time.monotonic() + 35
        queue_args = scheduler.parser().parse_args(['--output', str(output)])
        try:
            scheduler.run_loop(cfg, queue_args, lambda: time.monotonic() > deadline)
        finally:
            scheduler.ROOT, scheduler.RAMP_SECONDS, scheduler.POLL_SECONDS = old_root, old_ramp, old_poll
            scheduler.gpu_info, scheduler.gpu_processes, scheduler.available_ram = actual_gpu, actual_processes, actual_ram
        status = read_json(output / 'status.json')
        assert status['state'] == 'completed', status
        assert status['completed'] == status['total'] == 12
        assert status['max_observed_parallel'] == 10, status
        assert read_json(output / 'memory_corrections.json')
        combined = read_json(output / 'window_comparison.json')
        assert combined['complete'] and len(combined['groups']) == 48
        for tag in ('results', 'maps', 'features'):
            assert output.with_name(output.name + '_' + tag + '.zip').is_file()
        with zipfile.ZipFile(output.with_name(output.name + '_results.zip')) as archive:
            assert archive.testzip() is None
            assert output.name + '/table_4_3.csv' in archive.namelist()
            for relative in read_json(ROOT / 'FROZEN_SHA256.json'):
                assert output.name + '/source/frozen/' + relative in archive.namelist(), relative
        from dashboard import frame
        text, _, _ = frame(output)
        assert len(text.splitlines()) == 4
        print('[通过] 12个独立进程任务、10路硬上限、模拟OOM重试、四行界面、四窗口汇总和完整源码打包。', flush=True)
        assert before == {str(p): file_hash(p) for p in parent_root.rglob('*.json')}
    print('进程/协议集成检查通过；以上合成运行不产生正式分类结果。', flush=True)


if __name__ == '__main__':
    main()
