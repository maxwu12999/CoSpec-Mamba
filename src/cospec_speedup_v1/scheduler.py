#!/usr/bin/env python3
from bootstrap import ADDON,CORE,use_core,revision
use_core()
import scheduler as core_scheduler
import os
import subprocess
import sys
import time
from common import read_json,write_json,digest,SafeStop
from speed_runtime import metadata,record_source

original_prepare=core_scheduler.prepare
def prepare(args):
    config=original_prepare(args)
    config['runtime_speedup']=metadata()
    root=args.output.expanduser().resolve()
    write_json(root/'config.json',config)
    record_source(config)
    from results import complete
    if all(complete(config,j) for j in config['jobs']):return config
    key=digest(dict(gpu=config['gpu_identity'],revision=revision(),source=config['source_sha256'],
                    data=config['data_identity'],batch=config['train']['eval_batch'],
                    prefetch=config['runtime_speedup']['prefetch']))[:24]
    report=root/'runtime_speedup'/revision()/('benchmark_'+key+'.json')
    checked=read_json(report,{})
    if not checked.get('passed'):
        state=read_json(root/'status.json',{})
        state.update(state='preflight',phase='当前显卡上做新旧输入/预测对照与测速',active=0,active_jobs=[],updated_unix=time.time())
        write_json(root/'status.json',state)
        with (root/'speedup_benchmark.log').open('ab',buffering=0) as log:
            result=subprocess.run([sys.executable,'-u',str(ADDON/'benchmark.py'),'--config',str(root/'config.json'),
                                   '--report',str(report)],stdout=log,stderr=subprocess.STDOUT,
                                  env=core_scheduler.environment(args.gpu))
        if result.returncode==130 or (root/'STOP').exists():raise SafeStop()
        checked=read_json(report,{})
        if result.returncode or not checked.get('passed'):
            checked=dict(passed=False,enabled=False,reason='GPU comparison failed; retained original worker',
                         log=str(root/'speedup_benchmark.log'))
            write_json(root/'runtime_speedup'/'fallback.json',checked)
    config['runtime_speedup'].update(enabled=bool(checked.get('enabled')),benchmark=checked)
    write_json(root/'config.json',config)
    environment=read_json(root/'environment.json',{})
    environment['runtime_speedup']=config['runtime_speedup'];write_json(root/'environment.json',environment)
    if not checked.get('enabled'):
        core_scheduler.ROOT=CORE
        print('[加速回退] 样本未确认收益或兼容性未通过；继续原入口，已有任务保留。',flush=True)
    else:
        print('[加速启用] 当前卡新旧对照通过；测试样本倍率：'+str(round(checked['weighted_test_speedup'],2)),flush=True)
    return config

if __name__=='__main__':
    core_scheduler.ROOT=ADDON
    core_scheduler.prepare=prepare
    raise SystemExit(core_scheduler.main())
