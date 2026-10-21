"""Pure scheduling decisions, independently testable without an NVIDIA GPU."""
import math
from common import profile_key, batch_count


def reservation(profile, phase='initializing', multiplier=1.):
    if phase in ('maps','completed'):
        return 640. * multiplier
    if phase in ('test','features'):
        peak = profile['eval_peak_reserved_mib']
    else:
        peak = max(profile['train_peak_reserved_mib'], profile['eval_peak_reserved_mib'])
    return max(1024., peak * 1.25 + 640.) * multiplier


def can_admit(profile, active, profiles, gpu, headroom=1024., multiplier=1., available_ram=None):
    need = reservation(profile, multiplier=multiplier)
    unrealized = 0.
    for entry in active:
        p = profiles[entry['profile_key']]
        target = reservation(p, entry.get('phase','initializing'), entry.get('multiplier',1.))
        # Context + loaded kernels are counted by nvidia-smi. If unavailable,
        # reserve the full future peak instead of guessing it is already paid.
        actual = max(0., entry.get('actual_gpu_mib', 0.))
        unrealized += max(0., target - actual)
    if gpu['free_mib'] < need + unrealized + headroom:
        return False
    if available_ram is not None and available_ram < profile.get('rss_peak_mib',512)*1.2 + 1024:
        return False
    return True


def requires_exclusive(profile, multiplier, headroom, total_mib):
    """A conservative OOM correction must not strand a resumable task forever."""
    return reservation(profile, multiplier=multiplier) + headroom > total_mib


def estimated_work(config, job, profile):
    data=config['data'][job['dataset']]
    train_steps=batch_count(data['train_count'],config['train']['batch'])*config['train']['epochs']
    test_steps=math.ceil(data['test_count']/config['train']['eval_batch'])*16
    train=train_steps*profile['train_batch_seconds']
    test=test_steps*profile['eval_batch_seconds']
    exports=max(2.,test*.02) if job['seed']==1 else .1
    return dict(train=train,test=test,exports=exports,total=train+test+exports,
                train_steps=train_steps,test_steps=test_steps)


def work_done(cost, progress, finished=False):
    if finished:
        return cost['total']
    phase=progress.get('phase','initializing')
    train=min(1.,max(0.,progress.get('train_steps_done',0)/max(1,cost['train_steps'])))*cost['train']
    test=min(1.,max(0.,progress.get('test_steps_done',0)/max(1,cost['test_steps'])))*cost['test']
    if phase in ('test','features','maps','completed'):
        train=cost['train']
    if phase in ('features','maps','completed'):
        test=cost['test']
    exports=cost['exports']*min(1.,max(0.,progress.get('export_done',0))) if phase in ('features','maps','completed') else 0.
    return min(cost['total'],train+test+exports)


class ThroughputETA:
    def __init__(self):
        self.previous=None
        self.rate=None

    def update(self, now, done_units, remaining_units):
        if self.previous:
            elapsed=now-self.previous[0]
            delta=done_units-self.previous[1]
            if elapsed>=5 and delta>0:
                measured=delta/elapsed
                self.rate=measured if self.rate is None else .2*measured+.8*self.rate
                self.previous=(now,done_units)
        else:
            self.previous=(now,done_units)
        # No division by nominal slot count: jobs share a single GPU. The rate
        # is actual aggregate progress over wall time under current contention.
        return remaining_units/self.rate if self.rate and self.rate>0 else None
