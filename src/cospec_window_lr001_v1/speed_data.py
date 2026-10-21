"""Exactly the old pixels, rotations, labels and private RNG stream, in batches."""
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import torch

LOCAL=threading.local()
ORIGINAL_BATCH=None
ORIGINAL_INIT=None
ORIGINAL_MERGED=None
PREFETCH=True


class Prepared(list):
    pass


def vector_batch(self,indices):
    if isinstance(indices,Prepared):
        value=indices.value
        indices.value=None
        return value
    index=np.asarray(list(indices),dtype=np.int64)
    coordinates=self.coords[index]
    rr=self.rows[coordinates[:,0]][:,:,None]
    cc=self.cols[coordinates[:,1]][:,None,:]
    block=self.cube[rr,cc,:]
    if self.rng is None:
        xs=np.ascontiguousarray(block.transpose(0,3,1,2))
    else:
        # Scalar draws retain NumPy's exact cached-bit state, also for odd batches.
        rotations=np.array([int(self.rng.integers(4)) for _ in index],dtype=np.int8)
        xs=np.empty((len(index),block.shape[-1],block.shape[1],block.shape[2]),dtype=block.dtype)
        for angle in range(4):
            selected=np.flatnonzero(rotations==angle)
            if len(selected):
                values=np.rot90(block[selected],angle,axes=(1,2))
                xs[selected]=values.transpose(0,3,1,2)
    labels=np.ascontiguousarray(self.labels[index],dtype=np.int64)
    return torch.from_numpy(xs),torch.from_numpy(labels)


def initialized(self,meta,part,patch,seed=None,coordinates=None):
    ORIGINAL_INIT(self,meta,part,patch,seed,coordinates)
    if part=='tr':
        LOCAL.training_dataset=self


def prepare_batches(dataset,parts,prefetch=True):
    if not prefetch:
        for part in parts:
            ready=Prepared(part);ready.value=vector_batch(dataset,part);yield ready
        return
    iterator=iter(parts)
    try:first=next(iterator)
    except StopIteration:return
    # At most one batch ahead; no next-epoch data or RNG is prepared early.
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='cospec_input') as executor:
        current=first
        future=executor.submit(vector_batch,dataset,current)
        while True:
            value=future.result()
            try:following=next(iterator)
            except StopIteration:following=None
            next_future=executor.submit(vector_batch,dataset,following) if following is not None else None
            ready=Prepared(current);ready.value=value
            yield ready
            if following is None:break
            current,future=following,next_future


def merged(count,batch,generator):
    parts=ORIGINAL_MERGED(count,batch,generator)
    dataset=getattr(LOCAL,'training_dataset',None)
    if dataset is None or len(dataset)!=count:
        return parts
    return prepare_batches(dataset,parts,PREFETCH)


def install(prefetch=True):
    global ORIGINAL_BATCH,ORIGINAL_INIT,ORIGINAL_MERGED,PREFETCH
    import data,worker
    PREFETCH=prefetch
    if ORIGINAL_BATCH is not None:return
    ORIGINAL_BATCH=data.Patches.batch
    ORIGINAL_INIT=data.Patches.__init__
    ORIGINAL_MERGED=worker.merged_batches
    data.Patches.batch=vector_batch
    data.Patches.__init__=initialized
    worker.merged_batches=merged


def uninstall():
    global ORIGINAL_BATCH,ORIGINAL_INIT,ORIGINAL_MERGED
    if ORIGINAL_BATCH is None:return
    import data,worker
    data.Patches.batch=ORIGINAL_BATCH
    data.Patches.__init__=ORIGINAL_INIT
    worker.merged_batches=ORIGINAL_MERGED
    LOCAL.training_dataset=None
    ORIGINAL_BATCH=ORIGINAL_INIT=ORIGINAL_MERGED=None
