"""Existing CoSpec backbone and byte-verified original ConvMamba architecture."""
import ast
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from suite_common import ROOT, file_hash
from omamba.structure_model import StructureMambaV2, scan_permutations
from omamba.encoder_comparison import MaskAwareLinearEncoder
from omamba.evidence_mamba import EvidenceMamba

SPECS = {
 'A':dict(encoder='linear',augmentation=False,scan='snake4',structure=False),
 'B':dict(encoder='linear',augmentation=True,scan='snake4',structure=False),
 'C':dict(encoder='operator',augmentation=False,scan='snake4',structure=False),
 'D':dict(encoder='operator',augmentation=True,scan='snake4',structure=False),
 'D_structure':dict(encoder='operator',augmentation=True,scan='snake4',structure=True),
 'D_row':dict(encoder='operator',augmentation=True,scan='row',structure=False),
 'D_raster4':dict(encoder='operator',augmentation=True,scan='raster4',structure=False),
 'D_random4':dict(encoder='operator',augmentation=True,scan='random4',structure=False),
 'ConvMamba_aug':dict(encoder='original_convolution',augmentation=True,scan='original',structure=False),
}

class CoSpec(StructureMambaV2):
    def __init__(self,wave,stats,spec,seed,patch=15,dim=64,depth=4,basis=64,backend='cuda'):
        super().__init__(wave,stats['means'],stats['variances'],variant='scan_structure' if spec['structure'] else 'scan',
            initialization_seed=seed,patch_size=patch,dim=dim,depth=depth,basis_size=basis,backend=backend)
        self.spec=spec
        if spec['encoder']=='linear':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+707)
                self.operator=MaskAwareLinearEncoder(len(wave),self.dim)
        row=torch.arange(patch*patch); col=row.reshape(patch,patch).T.flatten()
        if spec['scan']=='row':orders=row.repeat(4,1)
        elif spec['scan']=='raster4':orders=torch.stack((row,row.flip(0),col,col.flip(0)))
        elif spec['scan']=='random4':
            g=torch.Generator().manual_seed(20261015)
            orders=torch.stack([row[torch.randperm(len(row),generator=g)] for _ in range(4)])
        else:orders=self.scan_orders
        self.scan_orders=orders;self.scan_inverse=orders.argsort(-1)
    def forward(self,patch,observed):
        spectra=patch.flatten(2).transpose(1,2)
        clean=torch.where(observed[:,None,:],spectra,torch.zeros_like(spectra))
        encoded=self.operator(clean,observed);features=encoded['features']
        tokens=features+self.position_encoder(self.positions)[None]
        reliability=encoded.get('relative_geometry_support')
        cue=self.structure_cue(clean,features,observed,reliability) if self.structure_cue is not None else None
        for i,block in enumerate(self.layers):
            order,inverse=self.scan_orders[i%4],self.scan_inverse[i%4]
            sequence=tokens.index_select(1,order);strength=None
            if self.spec['structure']:
                gap=self.structure_cue.transition_gap(clean,features,observed,order)
                importance=cue['importance'].index_select(1,order)/1.5
                strength=3*self.raw_forget_gain.sigmoid()*reliability[:,None]*importance*gap
            tokens=block(sequence,torch.ones_like(sequence[:,:,0]),strength).index_select(1,inverse)
        center=tokens[:,tokens.shape[1]//2]
        pooled=torch.einsum('s,bsd->bd',self.pool_logits.softmax(0),tokens)
        embedding=self.readout(torch.cat((center,pooled),-1))
        return {'logits':self.classifier(embedding),'embedding':embedding}

EXPECTED = {'ConvMamba.py':'05856dda301a486b3247db621cfd7622f647f2847fbd88f1f778e73dc752282f',
 'pipeline.py':'de0f671ed2ee734d7899b8e70b4b1248a21ce27fc6afba194e2f654a284a704c'}

class ReferenceMamba(EvidenceMamba):
    """Real Mamba-1 recurrence for CPU checks; never used by formal CUDA runs."""
    def __init__(self,d_model,d_state=16,d_conv=4,expand=2,**kwargs):
        super().__init__(d_model,d_state,d_conv,expand,backend='reference')

def original_classes(backend):
    if backend=='cuda':
        from mamba_ssm import Mamba
    else:Mamba=ReferenceMamba
    def rearrange(x,pattern):
        if pattern!='b s w h-> b (w h) s':raise ValueError(pattern)
        return x.flatten(2).transpose(1,2)
    namespace=dict(torch=torch,nn=nn,math=math,np=np,F=F,Mamba=Mamba,
                   rearrange=rearrange,init=nn.init)
    for name in ('ConvMamba.py','pipeline.py'):
        p=ROOT/'vendor'/'convmamba'/name
        if file_hash(p)!=EXPECTED[name]:raise ValueError(f'原版ConvMamba源码校验失败：{p}')
        tree=ast.parse(p.read_text(encoding='utf-8'))
        # Remove dependency-only imports (including unused torchvision/PIL).
        # All class/function bodies and initialization statements remain exact.
        tree.body=[node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
        exec(compile(tree,str(p),'exec'),namespace)
    return namespace

class ConvAdapter(nn.Module):
    def __init__(self,channels,classes,patch,depth,backend):
        super().__init__()
        params=dict(data=dict(num_classes=classes,patch_size=patch,spectral_size=channels),
          net=dict(dim=64,depth=depth,kernal=5,padding=2),mamba=dict(d_state=24,d_conv=4,expand=1))
        self.net=original_classes(backend)['Classifier'](params)
    def forward(self,x,observed):
        x=torch.where(observed[:,:,None,None],x,torch.zeros_like(x))
        embedding,_=self.net.encoder_block(x)
        return dict(logits=self.net.classifier_mlp(embedding),embedding=embedding)
    def component_counts(self):return dict(total=sum(p.numel() for p in self.parameters()))

def build_model(model_id,wave,stats,seed,options):
    spec=SPECS[model_id]
    if model_id.startswith('ConvMamba'):
        return ConvAdapter(len(wave),len(stats['means']),options['patch'],options['depth'],options['backend'])
    return CoSpec(wave,stats,spec,seed,**options)
