"""Explicit runtime extension, with complete source provenance in result ZIPs."""
import copy
import os
from pathlib import Path
from bootstrap import ADDON,revision


ORIGINAL_EVALUATE=None


def metadata(prefetch=None):
    if prefetch is None:prefetch=os.environ.get('COSPEC_PREFETCH','1')!='0'
    return dict(kind='cospec_speedup_v1',revision=revision(),prefetch=bool(prefetch),
                training='unchanged optimizer/math/order; exact batch assembly plus one CPU batch ahead',
                testing='unchanged batch/masks/metrics; input tiles reused across cases')


def record_source(config):
    from common import write_json
    info=metadata(config.get('runtime_speedup',{}).get('prefetch',True))
    source={p.name:p.read_text(encoding='utf-8') for p in sorted(ADDON.glob('*')) if p.suffix in ('.py','.sh','.json','.md')}
    write_json(Path(config['output'])/'runtime_speedup'/info['revision']/'source.json',dict(metadata=info,files=source))


def install(config=None,prefetch=True):
    global ORIGINAL_EVALUATE
    import worker
    import speed_data,speed_eval
    speed_data.install(prefetch)
    if ORIGINAL_EVALUATE is None:ORIGINAL_EVALUATE=worker.evaluate
    worker.evaluate=speed_eval.evaluate
    if not hasattr(worker,'_speed_original_write_json'):
        worker._speed_original_write_json=worker.write_json
        def write(path,value):
            if Path(path).name=='result.json' and isinstance(value,dict) and value.get('complete'):
                value=copy.deepcopy(value);value.setdefault('runtime',{})['speedup']=metadata(speed_data.PREFETCH)
            return worker._speed_original_write_json(path,value)
        worker.write_json=write
    if config is not None:record_source(config)


def uninstall():
    global ORIGINAL_EVALUATE
    import worker,speed_data
    speed_data.uninstall()
    if ORIGINAL_EVALUATE is not None:
        worker.evaluate=ORIGINAL_EVALUATE;ORIGINAL_EVALUATE=None
    if hasattr(worker,'_speed_original_write_json'):
        worker.write_json=worker._speed_original_write_json
        del worker._speed_original_write_json
