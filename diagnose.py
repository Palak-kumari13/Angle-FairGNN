"""Measure whether task and sensitive interventions change the same edges.

Task directions: per-node gradients of train-label CE with respect to the edge mask.
Sensitive directions: discrete sensitive feature flips (if encoded in X), and/or
probe-guided proxy-feature interventions. Neither is a causal counterfactual.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from train import MaskGCN, load_npz, undirected_edges
from src.subspaces import basis, angles, null_overlaps, selective_basis, whitened_angles


def sample_drift(model, x, edges, indices, sensitive_col=None, proxy=False, step=0.5):
    model.eval()
    with torch.no_grad():
        baseline = model(x, edges)[1].detach()
    drift = []
    if proxy:
        features = x[indices].detach().cpu().numpy()
        # The probe is fitted on TRAIN indices only.
        raise RuntimeError('Proxy vector is computed in main() using the training split')
    if sensitive_col is None:
        return np.empty((0, baseline.numel()))
    for i in indices.tolist():
        changed = x.clone()
        changed[i, sensitive_col] = 1.0 - x[i, sensitive_col]
        with torch.no_grad():
            updated = model(changed, edges)[1]
        drift.append((updated-baseline).detach().cpu().numpy())
    return np.asarray(drift)


def proxy_drift(model, x, edges, indices, sensitive, train, step):
    estimator = LogisticRegression(max_iter=1000)
    estimator.fit(x[train].detach().cpu().numpy(), sensitive[train].cpu().numpy())
    direction = torch.tensor(estimator.coef_.reshape(-1), dtype=x.dtype, device=x.device)
    direction = direction / direction.norm().clamp_min(1e-8)
    with torch.no_grad():
        original = model(x, edges)[1].detach()
    samples=[]
    for i in indices.tolist():
        perturb=x.clone();perturb[i] += step * direction
        with torch.no_grad():
            samples.append((model(perturb,edges)[1]-original).cpu().numpy())
    return np.asarray(samples)


def task_drift(model,x,edges,y,indices):
    model.eval()
    samples=[]
    for i in indices.tolist():
        mask=model.get_mask(x,edges)
        # Gradient of true-label task loss w.r.t. the mask, keeping model fixed.
        logits,_=model(x,edges,mask_override=mask)
        loss=F.cross_entropy(logits[i:i+1],y[i:i+1])
        gradient=torch.autograd.grad(loss, mask, retain_graph=False)[0]
        samples.append(gradient.detach().cpu().numpy())
    return np.asarray(samples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--sensitive-col',type=int,default=None,help='Binary sensitive feature column; omit if unavailable')
    p.add_argument('--proxy',action='store_true',help='Add train-only sensitive-probe direction as a sensitivity test')
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--dropout',type=float,default=.3)
    p.add_argument('--samples',type=int,default=24);p.add_argument('--rank',type=int,default=8)
    p.add_argument('--proxy-step',type=float,default=.5)
    p.add_argument('--variance',type=float,default=.95);p.add_argument('--null-repeats',type=int,default=100)
    p.add_argument('--bootstrap',type=int,default=100);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--min-drift',type=float,default=1e-7);p.add_argument('--output',default='results/diagnostic')
    args=p.parse_args()
    if args.sensitive_col is None and not args.proxy:
        p.error('Specify --sensitive-col or --proxy (or both)')
    device=torch.device('cpu')
    x,edge_raw,y,s,idx=load_npz(args.data,device)
    edges=undirected_edges(edge_raw,len(x))
    model=MaskGCN(x.shape[1],args.hidden,args.dropout,masked=True)
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
    model.eval()
    rng=np.random.default_rng(args.seed)
    tr=idx['train_idx']
    selected=tr[torch.from_numpy(rng.choice(len(tr),size=min(args.samples,len(tr)),replace=False))]
    with torch.no_grad():
        m=model.get_mask(x,edges)
    near_zero=float((m<.1).float().mean());near_one=float((m>.9).float().mean())
    a_parts=[]
    if args.sensitive_col is not None:
        if not 0<=args.sensitive_col<x.size(1):p.error('Invalid --sensitive-col')
        a_parts.append(sample_drift(model,x,edges,selected,args.sensitive_col))
    if args.proxy:
        a_parts.append(proxy_drift(model,x,edges,selected,s,tr,step=args.proxy_step))
    A=np.vstack(a_parts)
    T=task_drift(model,x,edges,y,selected)
    drift_a=float(np.linalg.norm(A,axis=1).mean());drift_t=float(np.linalg.norm(T,axis=1).mean())
    mask_std=float(m.std(unbiased=False))
    status='ok'
    if near_zero>.95 or near_one>.95 or mask_std<1e-4:status='collapsed_mask'
    if drift_a<args.min_drift or drift_t<args.min_drift:status='negligible_drift'
    ua,sa=basis(A,args.variance,args.rank)
    ut,st=basis(T,args.variance,args.rank)
    measured,overlap=angles(ua,ut)
    whitened,whiten_overlap=whitened_angles(A,T,variance=args.variance,max_rank=args.rank)
    null=null_overlaps(ua,ut,args.null_repeats,args.seed) if ua.shape[1] and ut.shape[1] else np.array([])
    qb=selective_basis(ua,ut)
    boots=[]
    if status=='ok':
        for _ in range(args.bootstrap):
            aa=A[rng.integers(len(A),size=len(A))]
            tt=T[rng.integers(len(T),size=len(T))]
            ua_b,_=basis(aa,args.variance,args.rank);ut_b,_=basis(tt,args.variance,args.rank)
            boots.append(angles(ua_b,ut_b)[1])
    result={'status':status,'n_train_samples':len(selected),'ambient_edges':int(edges.shape[1]),
            'mask_mean':float(m.mean()),'mask_std':mask_std,'mask_below_01':near_zero,'mask_above_09':near_one,
            'sensitive_drift_norm':drift_a,'task_gradient_norm':drift_t,
            'sensitive_rank':int(ua.shape[1]),'task_rank':int(ut.shape[1]),'selective_rank':int(qb.shape[1]),
            'angles_degrees':measured.tolist(),'overlap':overlap,
            'whitened_angles_degrees':whitened.tolist(),'whitened_overlap':whiten_overlap,
            'null_mean':float(null.mean()) if len(null) else None,
            'null_p_ge_observed':float((np.sum(null>=overlap)+1)/(len(null)+1)) if len(null) else None,
            'bootstrap_overlap_95':np.quantile(boots,[.025,.975]).tolist() if len(boots) else None,
            'warning':'intervention sensitivity only; not causal fairness'}
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez_compressed(folder/'subspaces.npz',sensitive_basis=ua,task_basis=ut,selective_basis=qb,
                        sensitive_singular=sa,task_singular=st,null_overlaps=null)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
