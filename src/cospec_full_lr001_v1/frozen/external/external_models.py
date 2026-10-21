"""Adapters around pinned upstream networks, without executing training scripts."""
import ast
import collections
from dataclasses import dataclass
from functools import partial
import math
import sys
import types
from typing import Union
import numpy as np
import torch
from torch import nn, Tensor, einsum
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from external_common import ROOT, read_json, file_hash

_VERIFIED = False

def verify_vendor():
    global _VERIFIED
    if _VERIFIED: return
    for source in read_json(ROOT / 'UPSTREAM.json').values():
        for entry in source['files']:
            if file_hash(ROOT / entry['local']) != entry['sha256']:
                raise ValueError('外部模型源码已变动：' + entry['local'])
    _VERIFIED = True

class _Definitions(ast.NodeTransformer):
    def visit_Expr(self, node):
        # MiM's unmodified dependency prints entire activation tensors in every
        # forward pass. Removing only print statements preserves all arithmetic.
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
            return None
        return self.generic_visit(node)

class _ScanOutputDimensions(ast.NodeTransformer):
    """Keep batch/sequence axes when the upstream scan contracts its state."""
    def __init__(self): self.replacements = 0
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'squeeze'
                and not node.args and not node.keywords):
            node.args = [ast.Constant(value=-1)]
            self.replacements += 1
        return node

def definitions(relative, bindings=None, assignments=(), remove_debug=False, scan_output_dims=False):
    verify_vendor()
    path = ROOT / 'vendor' / relative
    name = '_cospec_external_' + relative.replace('/', '_').replace('.', '_')
    module = types.ModuleType(name); sys.modules[name] = module
    module.__dict__.update(torch=torch, nn=nn, F=F, np=np, math=math, Tensor=Tensor,
        einsum=einsum, rearrange=rearrange, repeat=repeat, Rearrange=Rearrange,
        Reduce=Reduce, partial=partial, dataclass=dataclass, Union=Union,
        collections=collections, init=nn.init)
    if bindings: module.__dict__.update(bindings)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in node.targets):
            nodes.append(node)
    tree.body = nodes
    if remove_debug: tree = _Definitions().visit(tree)
    if scan_output_dims:
        repair = _ScanOutputDimensions()
        tree = repair.visit(tree)
        if repair.replacements != 2:
            raise ValueError('Expected exactly two upstream scan output squeezes')
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module.__dict__

def mim_namespace(scan_backend='native'):
    parallel = definitions('mim_dependencies/pscan.py', assignments=('pscan',))['pscan']
    zeta_parallel = definitions('mim_dependencies/zeta_pscan.py', assignments=('pscan',))['pscan']
    scans = definitions('mim_dependencies/ssm.py', {'pscan': zeta_parallel}, scan_output_dims=True)
    native_scan = scans['selective_scan']
    if scan_backend == 'cuda_fused':
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        def fused(x, delta, A, B, C, D):
            # zeta applies softplus before this function, so it is disabled here.
            return selective_scan_fn(x.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(),
                A, B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(), D,
                delta_softplus=False).transpose(1, 2)
        scans['selective_scan'] = fused
    elif scan_backend != 'native': raise ValueError(scan_backend)
    vision = definitions('mim_dependencies/vision_model.py', {'SSM': scans['SSM']}, remove_debug=True)
    scan_order = definitions('mim/multiscan_v3.py')
    namespace = definitions('mim/MiM_v2.py', dict(pscan=parallel,
        snake_flatten=scan_order['snake_flatten'], snake_unflatten=scan_order['snake_unflatten'],
        vim=vision['VisionEncoderMambaBlock']))
    return namespace, scans, native_scan

def model_settings(model, dataset):
    if model == 'HybridSN': return dict(pca_components=30, pca_whiten=True)
    if model == 'A2S2K': return dict(kernel_channels=24, reduction=2, global_pool='adaptive_average')
    if model == 'SQSFormer':
        return dict(dim=64, depth=3, heads=20, kernal=5, padding=2, mlp_dim=8, dropout=0.)
    if model == 'GSCViT':
        heads = 16 if dataset == 'IndianPines' else (4 if dataset in ('HongHu', 'HanChuan', 'LongKou') else 1)
        return dict(heads=[heads]*3, depth=[1]*3, group_spatial_size=[4]*3,
                    dropout=.1, padding=[1]*3, dims=[256,128,64],
                    num_groups=[8 if dataset == 'IndianPines' else 16]*3,
                    input_extension='reflect right/bottom to multiple of four; no additional scene pixels')
    if model == 'MiM_v2': return dict(dim=128, depth=1, emb_dropout=0., patch_size=1, upstream_variant='MiM-v2')
    raise KeyError(model)

class ExternalModel(nn.Module):
    def __init__(self, model, dataset, channels, classes, patch=15, pca=None, scan_backend='native'):
        super().__init__()
        self.model_name = model; self.channels = channels; self.patch = patch
        self.settings = model_settings(model, dataset)
        self._embedding = None
        if model == 'HybridSN':
            if pca is None: raise ValueError('HybridSN requires training-only PCA')
            self.register_buffer('pca_mean', torch.as_tensor(pca['mean'], dtype=torch.float32))
            self.register_buffer('pca_projection', torch.as_tensor(pca['projection'], dtype=torch.float32))
            native = definitions('hybridsn/HybridSN.py')
            self.net = native['HybridSN'](len(self.pca_projection), patch, classes)
            head = self.net.dense3[-1]
        elif model == 'A2S2K':
            native = definitions('a2s2k/A2S2KResNet.py', {'PARAM_KERNEL_SIZE': 24})
            self.net = native['S3KAIResNet'](channels, classes, 2)
            # Original AvgPool3d(5,5,1) is global for a 7x7 input. Keep global
            # pooling when extending to 15x15; learned weights are unchanged.
            self.net.avg_pooling = nn.AdaptiveAvgPool3d(1)
            head = self.net.full_connection[-1]
        elif model == 'SQSFormer':
            native = definitions('sqsformer/SQSFormer.py')
            params = dict(data=dict(num_classes=classes, patch_size=patch, spectral_size=channels), net=self.settings)
            self.net = native['SQSFormer'](params)
            head = self.net.classifier_mlp[-1]
        elif model == 'GSCViT':
            native = definitions('gscvit/gscvit.py')
            kwargs = {k:v for k,v in self.settings.items() if k != 'input_extension'}
            kwargs['heads'] = tuple(kwargs['heads'])
            self.net = native['GSCViT'](num_classes=classes, channels=channels, **kwargs)
            head = self.net.mlp_head[-1]
        elif model == 'MiM_v2':
            native, _, _ = mim_namespace(scan_backend)
            self.net = native['MiM_v2'](channels, classes, patch, dim=128, depth=1, emb_dropout=0.)
            head = self.net.mlp_head[-1]
        else: raise KeyError(model)
        self._hook = head.register_forward_pre_hook(self._capture_embedding)

    def _capture_embedding(self, module, arguments):
        self._embedding = arguments[0].detach()

    def preprocess(self, x, observed):
        if x.ndim != 4 or observed.shape != x.shape[:2]: raise ValueError('Input/mask shape mismatch')
        x = torch.where(observed[:, :, None, None], x, torch.zeros_like(x))
        if self.model_name == 'HybridSN':
            x = torch.einsum('bchw,kc->bkhw', x - self.pca_mean[None,:,None,None], self.pca_projection)
        elif self.model_name == 'A2S2K':
            x = x.permute(0,2,3,1).unsqueeze(1)
        elif self.model_name == 'GSCViT':
            extra_h, extra_w = (-x.shape[-2]) % 4, (-x.shape[-1]) % 4
            if extra_h or extra_w: x = F.pad(x, (0,extra_w,0,extra_h), mode='reflect')
        elif self.model_name == 'MiM_v2':
            x = x.unsqueeze(1)
        return x

    def forward(self, x, observed):
        output = self.net(self.preprocess(x, observed))
        logits = output[0] if self.model_name == 'SQSFormer' else output
        return dict(logits=logits, embedding=self._embedding)

    def component_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad))
