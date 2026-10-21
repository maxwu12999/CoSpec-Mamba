"""Retain checkpoint identities only for an explicitly reviewed runtime repair.

The legacy fixed-0.001 release is supported; the old staged-LR experiments are
not. Scientific settings, data, masks, model files, and training/evaluation
function bodies must remain identical before a legacy identity can be retained.
"""
import ast
import copy
from common import ROOT,read_json,file_hash,digest


def scientific_guard():
    manifest=read_json(ROOT/'COMPATIBILITY.json')
    for relative,expected in manifest['scientific_files'].items():
        if file_hash(ROOT/relative)!=expected:
            raise ValueError('更新改变了科学实现，不能沿用已有检查点：'+relative)
    for relative,expected_nodes in manifest['scientific_nodes'].items():
        tree=ast.parse((ROOT/relative).read_text(encoding='utf-8'))
        nodes={n.name:digest(ast.dump(n,include_attributes=False)) for n in tree.body if hasattr(n,'name')}
        for name,expected in expected_nodes.items():
            if nodes.get(name)!=expected:
                raise ValueError('更新改变了训练/评估函数，不能沿用已有检查点：'+relative+':'+name)
    return manifest


def resolve_protocol(protocol,old=None):
    if old is None or old.get('source_sha256')==protocol['source_sha256']:
        return protocol,dict(used=False)
    manifest=read_json(ROOT/'COMPATIBILITY.json')
    if old.get('source_sha256') not in manifest['reviewed_legacy_source_sha256']:
        return protocol,dict(used=False)
    scientific_guard()
    candidate=copy.deepcopy(protocol)
    candidate['source_sha256']=old['source_sha256']
    if digest(candidate)!=old.get('protocol_id'):
        # A different LR, batch, job list, data/mask hash, etc. is not an update.
        return protocol,dict(used=False)
    return candidate,dict(used=True,reason='reviewed scheduling and preflight repair; scientific implementation unchanged',
                         inherited_source_sha256=old['source_sha256'],runtime_source_sha256=protocol['source_sha256'],
                         guard_manifest_sha256=file_hash(ROOT/'COMPATIBILITY.json'),
                         results_and_checkpoints_rewritten=False)
