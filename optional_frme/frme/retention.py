"""Standalone continual-learning utilities; no federated aggregation."""
import copy
import torch

def freeze_teacher(model):
    teacher=copy.deepcopy(model).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    return teacher

def replay_memory(data,cap=48,seed=0):
    gen=torch.Generator(device='cpu').manual_seed(seed)
    idx=data['train'][torch.randperm(len(data['train']),generator=gen)[:cap]]
    return {k:data[k].clone() for k in ('x','edges','y')} | {'idx':idx.clone()}
