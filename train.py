"""Standalone AngleFair-GNN training: A0 reference, A1 mask, A2 fidelity, A3 fairness, A4 robustness, A5 distillation."""
import argparse, csv, json, random, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score
from robustness import edit_edges, aligned_mask_drift, evaluate_drift
from global_fallback import choose_checkpoint, GateConfig


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_npz(path, device):
    arr=np.load(path,allow_pickle=False)
    required=['x','edge_index','y','sensitive','train_idx','val_idx','test_idx']
    missing=[k for k in required if k not in arr]
    if missing: raise ValueError(f'{path}: missing {missing}')
    x=torch.as_tensor(arr['x'],dtype=torch.float32,device=device)
    ei=torch.as_tensor(arr['edge_index'],dtype=torch.long,device=device)
    y=torch.as_tensor(arr['y'],dtype=torch.long,device=device).view(-1)
    s=torch.as_tensor(arr['sensitive'],dtype=torch.long,device=device).view(-1)
    if ei.shape[0]!=2 and ei.shape[1]==2: ei=ei.T.contiguous()
    assert ei.shape[0]==2 and ei.numel()>0
    n=len(x)
    if x.ndim != 2 or not torch.isfinite(x).all(): raise ValueError('x must be a finite [N,F] matrix')
    if len(y)!=n or len(s)!=n: raise ValueError('labels/sensitive must match node count')
    if set(y.unique().tolist()) != {0, 1}: raise ValueError('both task labels must be present')
    if ei.min()<0 or ei.max()>=n: raise ValueError('edge indices out of range')
    if set(y.unique().tolist())-set([0,1]): raise ValueError('binary y required')
    if set(s.unique().tolist())-set([0,1]): raise ValueError('binary sensitive required')
    idx={k:torch.as_tensor(arr[k],dtype=torch.long,device=device).view(-1) for k in ['train_idx','val_idx','test_idx']}
    for k,v in idx.items():
        if v.numel()==0 or v.min()<0 or v.max()>=n or len(v.unique())!=len(v): raise ValueError('bad '+k)
    sets=[set(v.cpu().tolist()) for v in idx.values()]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise ValueError('splits overlap')
    return x,ei,y,s,idx


def undirected_edges(edge_index, num_nodes):
    a,b=edge_index
    keep=a!=b
    a,b=a[keep],b[keep]
    lo,hi=torch.minimum(a,b),torch.maximum(a,b)
    pairs=torch.unique(torch.stack((lo,hi),dim=1),dim=0)
    return pairs.T.contiguous()


class GCNLayer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__(); self.lin=nn.Linear(in_dim,out_dim)
    def forward(self,x,edge_index,weights):
        n=x.size(0); u,v=edge_index
        # Symmetric weighted GCN normalization; graph edges supplied once as undirected pairs.
        u2=torch.cat([u,v,torch.arange(n,device=x.device)])
        v2=torch.cat([v,u,torch.arange(n,device=x.device)])
        w=torch.cat([weights,weights,torch.ones(n,device=x.device)])
        deg=torch.zeros(n,device=x.device).index_add(0,v2,w)
        norm=w*(deg[u2].clamp_min(1e-8)*deg[v2].clamp_min(1e-8)).rsqrt()
        h=self.lin(x)
        out=torch.zeros_like(h).index_add(0,v2,h[u2]*norm[:,None])
        return out


class MaskGCN(nn.Module):
    def __init__(self,nfeatures,hidden,dropout=0.3,masked=False):
        super().__init__()
        self.masked=masked; self.dropout=dropout
        self.pre=nn.Linear(nfeatures,hidden)
        self.g1=GCNLayer(nfeatures,hidden)
        self.g2=GCNLayer(hidden,2)
        if masked:
            self.mask_mlp=nn.Sequential(nn.Linear(hidden*3,hidden),nn.ReLU(),nn.Linear(hidden,1))
            nn.init.zeros_(self.mask_mlp[-1].weight)
            nn.init.constant_(self.mask_mlp[-1].bias, 0.4)
    def get_mask(self,x,ei):
        if not self.masked:return torch.ones(ei.size(1),device=x.device)
        # Use local graph context so edge importance can respond to structural edits.
        u,v=ei
        h0=torch.relu(self.pre(x))
        agg=torch.zeros_like(h0).index_add(0,u,h0[v]).index_add(0,v,h0[u])
        deg=torch.zeros(x.size(0),device=x.device).index_add(0,u,torch.ones_like(u,dtype=h0.dtype)).index_add(0,v,torch.ones_like(v,dtype=h0.dtype))
        h=torch.relu(h0+agg/deg.clamp_min(1)[:,None])
        z=torch.cat([h[u]+h[v],(h[u]-h[v]).abs(),h[u]*h[v]],dim=1)
        return torch.sigmoid(self.mask_mlp(z).flatten())
    def forward(self,x,ei,force_full=False, mask_override=None):
        m=self.get_mask(x,ei)
        w=torch.ones_like(m) if force_full else (mask_override if mask_override is not None else m)
        h=F.relu(self.g1(x,ei,w));h=F.dropout(h,p=self.dropout,training=self.training)
        return self.g2(h,ei,w),m


def safe_auc(y,score):
    return float(roc_auc_score(y,score)) if len(np.unique(y))==2 else float('nan')


def metrics(logits,y,s,indices,threshold=0.5):
    pr=logits.softmax(-1)[:,1].detach().cpu().numpy()
    yt=y.detach().cpu().numpy(); st=s.detach().cpu().numpy()
    ids=indices.detach().cpu().numpy(); a=yt[ids]; b=(pr[ids]>=threshold).astype(int); sen=st[ids]
    rates=[]; tprs=[]
    for g in [0,1]:
        mask=sen==g
        rates.append(float(b[mask].mean()) if mask.any() else float('nan'))
        positive=mask&(a==1)
        tprs.append(float(b[positive].mean()) if positive.any() else float('nan'))
    return {'f1':float(f1_score(a,b,zero_division=0)),'auc':safe_auc(a,pr[ids]),
            'parity':abs(rates[0]-rates[1]),'equality':abs(tprs[0]-tprs[1]),
            'positive_prediction_rate':float(b.mean())}


def differentiable_fairness(logits, y, sensitive, train_idx):
    """Train-only soft DP and EO differences (binary positive-class probabilities)."""
    probabilities = logits.softmax(-1)[:, 1]
    labels, groups = y[train_idx], sensitive[train_idx]
    probs = probabilities[train_idx]
    g0, g1 = groups == 0, groups == 1
    if not (g0.any() and g1.any() and (g0 & (labels == 1)).any() and (g1 & (labels == 1)).any()):
        raise ValueError('A3 requires both sensitive groups and positive labels within both groups in TRAIN')
    dp = (probs[g0].mean() - probs[g1].mean()).square()
    eo = (probs[g0 & (labels == 1)].mean() - probs[g1 & (labels == 1)].mean()).square()
    return dp, eo


def mask_exposure_gap(mask, edge_index, sensitive, train_idx, num_nodes):
    """Degree-normalized incident edge importance gap on selected nodes.

    A group-level descriptive proxy, not a causal sensitive-direction projection.
    """
    u, v = edge_index
    sums = torch.zeros(num_nodes, device=mask.device).index_add(0, u, mask).index_add(0, v, mask)
    counts = torch.zeros(num_nodes, device=mask.device).index_add(0, u, torch.ones_like(mask)).index_add(0, v, torch.ones_like(mask))
    has_edges = counts > 0
    group0 = (sensitive == 0) & has_edges
    group1 = (sensitive == 1) & has_edges
    selected = torch.zeros(num_nodes, dtype=torch.bool, device=mask.device)
    selected[train_idx] = True
    group0, group1 = group0 & selected, group1 & selected
    if not group0.any() or not group1.any():
        raise ValueError('Need nodes with incident edges from both sensitive groups')
    exposure = sums / counts.clamp_min(1)
    return (exposure[group0].mean() - exposure[group1].mean()).abs()


def train_one(data,model_name,seed,args,device):
    set_seed(seed)
    x,ei_raw,y,s,idx=data
    ei=undirected_edges(ei_raw,len(x))
    model=MaskGCN(x.size(1),args.hidden,args.dropout,model_name!='A0').to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # A2/A3 use a frozen A0 reference trained on the same seed and split.
    teacher=None
    if model_name in ('A2', 'A3', 'A4', 'A5'):
        teacher=MaskGCN(x.size(1),args.hidden,args.dropout,False).to(device)
        ckpt=Path(args.output)/f'{args.dataset}_A0_seed{seed}.pt'
        if not ckpt.exists(): raise FileNotFoundError(f'Run A0 first; teacher missing: {ckpt}')
        teacher.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
        teacher.eval();teacher.requires_grad_(False)
        with torch.no_grad():
            teacher_logits = teacher(x,ei)[0]
            reference = teacher_logits.softmax(-1)
    best=-float('inf');best_state=None;bad=0
    max_mask_grad=0.0
    for epoch in range(args.epochs):
        model.train();opt.zero_grad()
        warmup = model.masked and epoch < args.mask_warmup
        logits,m=model(x,ei,force_full=warmup)
        loss=F.cross_entropy(logits[idx['train_idx']],y[idx['train_idx']])
        if model_name in ('A2', 'A3', 'A4', 'A5'):
            # Distillation/reference fidelity from frozen full-graph A0 teacher, on TRAIN nodes only.
            pred=F.log_softmax(logits[idx['train_idx']],dim=-1)
            fidelity=F.kl_div(pred,reference[idx['train_idx']],reduction='batchmean')
            loss=loss+args.lambda_fid*fidelity
            if not warmup:
                loss=loss+args.lambda_budget*(m.mean()-args.retention)**2
        if model_name in ('A3', 'A4', 'A5'):
            dp, eo = differentiable_fairness(logits, y, s, idx['train_idx'])
            exposure = mask_exposure_gap(m, ei, s, idx['train_idx'], len(x))
            ramp = min(1.0, max(0.0, (epoch - args.mask_warmup + 1) / max(1, args.fair_ramp)))
            loss = loss + ramp*(args.lambda_dp*dp + args.lambda_eo*eo + args.lambda_mask_fair*exposure.square())
        if model_name in ('A4', 'A5') and args.lambda_robust > 0 and not warmup:
            # A4: prediction consistency on train nodes under a fixed-budget graph edit.
            # This does not establish adversarial robustness or explanation fidelity.
            perturbed = edit_edges(ei, len(x), args.attack_budget, seed * 100000 + epoch)
            logits_edit, mask_edit = model(x, perturbed)
            clean_distribution = logits[idx['train_idx']].detach().softmax(-1)
            loss = loss + args.lambda_robust * F.kl_div(
                F.log_softmax(logits_edit[idx['train_idx']], dim=-1),
                clean_distribution, reduction='batchmean')
            if model.masked and args.lambda_mask_stability > 0:
                loss = loss + args.lambda_mask_stability * aligned_mask_drift(
                    ei, m, perturbed, mask_edit)
        if model_name == 'A5' and args.lambda_distill > 0:
            # A5: stronger teacher-student retention on TRAIN nodes only.
            loss = loss + args.lambda_distill * F.kl_div(
                F.log_softmax(logits[idx['train_idx']], dim=-1),
                reference[idx['train_idx']], reduction='batchmean')
        loss.backward()
        if model.masked and not warmup and model.mask_mlp[-1].weight.grad is not None:
            max_mask_grad=max(max_mask_grad,float(model.mask_mlp[-1].weight.grad.norm().detach()))
        opt.step()
        model.eval()
        with torch.no_grad(): vl,_=model(x,ei);vmet=metrics(vl,y,s,idx['val_idx'])
        score=(vmet['auc']+vmet['f1'])/2 if np.isfinite(vmet['auc']) else vmet['f1']
        if not np.isfinite(score): raise ValueError('Validation score undefined; inspect labels and split')
        if warmup:
            continue
        if score>best+1e-7:
            best=score;bad=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=args.patience:break
    if best_state is None: raise RuntimeError('No post-warmup checkpoint; increase --epochs')
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        lg,mask=model(x,ei)
        val_scores=lg.softmax(-1)[idx['val_idx'],1].detach().cpu().numpy()
        val_labels=y[idx['val_idx']].detach().cpu().numpy()
        thresholds=np.linspace(0.1,0.9,81)
        threshold=float(max(thresholds,key=lambda t:(f1_score(val_labels,val_scores>=t,zero_division=0),-abs(t-.5))))
        out=metrics(lg,y,s,idx['test_idx'],threshold)
        out['validation_threshold']=threshold
        validation=metrics(lg,y,s,idx['val_idx'],threshold)
        out['val_auc']=validation['auc']
        out['val_f1']=validation['f1']
        out['val_parity']=validation['parity']
        out['val_equality']=validation['equality']
        out['val_mask_std']=float(mask.std(unbiased=False)) if model.masked else float('nan')
    if model_name != 'A0':
        with torch.no_grad():
            out['mask_exposure_gap'] = float(mask_exposure_gap(mask,ei,s,idx['test_idx'],len(x)))
    else:
        out['mask_exposure_gap'] = float('nan')
    # All A0–A3 use the same held-out attack protocol and perturbation sequence.
    attack = evaluate_drift(model, x, ei, y, s, idx['test_idx'],
                            args.attack_budget, args.drift_steps, seed + 901)
    out.update(attack)
    # Independent perturbation sample; selection uses validation nodes only.
    val_attack=evaluate_drift(model,x,ei,y,s,idx['val_idx'],
                              args.attack_budget,args.drift_steps,seed + 1901)
    # evaluate_drift emits metric names depending on the attack implementation.
    # A missing stability metric causes conservative fallback, not silent acceptance.
    out['val_prediction_drift']=float(val_attack.get('drift_pred_change_max',float('nan')))
    out.update({'random_mask_f1':float('nan'),'mask_entropy':float('nan'),'mask_lt_01':float('nan'),'mask_gt_09':float('nan'),'mask_grad_max':float('nan'),'mask_score_std':float('nan'),'degenerate_mask':0})
    if model.masked:
        with torch.no_grad():
            random_mask=torch.rand(mask.shape,device=device)
            # Matched mean magnitude; random-mask sufficiency is a diagnostic, not a ground-truth explanation label.
            random_mask=(random_mask*mask.mean()/random_mask.mean().clamp_min(1e-8)).clamp(0,1)
            random_logits,_=model(x,ei,mask_override=random_mask)
            out['random_mask_f1']=metrics(random_logits,y,s,idx['test_idx'],threshold)['f1']
            out['mask_entropy']=float((-mask.clamp(1e-7,1-1e-7)*mask.clamp(1e-7,1-1e-7).log()-(1-mask).clamp(1e-7,1-1e-7)*(1-mask).clamp(1e-7,1-1e-7).log()).mean())
            out['mask_lt_01']=float((mask<.1).float().mean())
            out['mask_gt_09']=float((mask>.9).float().mean())
            out['mask_grad_max']=max_mask_grad
            out['mask_score_std']=float(mask.std(unbiased=False))
            out['degenerate_mask']=int(float((mask<.1).float().mean())>.95 or float((mask>.9).float().mean())>.95 or out['mask_score_std']<1e-4)
        if out['degenerate_mask']: print(f'WARNING: near-uniform mask in {args.dataset} {model_name} seed {seed}')
    out.update({'dataset':args.dataset,'model':model_name,'seed':seed,
        'mask_mean':float(mask.mean()),'mask_frac_gt_05':float((mask>0.5).float().mean()),
        'best_validation_score':float(best),'epochs_trained':epoch+1, 'stage':args.stage, 'retention_target':args.retention})
    torch.save(model.state_dict(),Path(args.output)/f'{args.dataset}_{model_name}_seed{seed}.pt')
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True);p.add_argument('--dataset',choices=['bail','credit','german'],required=True)
    p.add_argument('--output',default=None);p.add_argument('--stage',type=int,choices=[1,2,3,4],default=4)
    p.add_argument('--models',nargs='+',choices=['A0','A1','A2','A3','A4','A5'],default=None)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=40)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--dropout',type=float,default=.3)
    p.add_argument('--lr',type=float,default=.005);p.add_argument('--weight-decay',type=float,default=0.0005)
    p.add_argument('--lambda-dp',type=float,default=1.0);p.add_argument('--lambda-eo',type=float,default=1.0)
    p.add_argument('--lambda-mask-fair',type=float,default=0.1)
    p.add_argument('--lambda-fid',type=float,default=1.0)
    p.add_argument('--lambda-budget',type=float,default=1.0)
    p.add_argument('--lambda-robust',type=float,default=0.2)
    p.add_argument('--lambda-mask-stability',type=float,default=0.1)
    p.add_argument('--lambda-distill',type=float,default=0.2)
    p.add_argument('--retention',type=float,default=0.5)
    p.add_argument('--mask-warmup',type=int,default=15)
    p.add_argument('--fair-ramp',type=int,default=30)
    p.add_argument('--max-dp',type=float,default=0.15)
    p.add_argument('--max-eo',type=float,default=0.15)
    p.add_argument('--max-auc-drop',type=float,default=0.03)
    p.add_argument('--max-prediction-drift',type=float,default=0.08)
    p.add_argument('--attack-budget',type=int,default=5,help='Undirected edge edits per step')
    p.add_argument('--drift-steps',type=int,default=5)
    args=p.parse_args()
    if not 0<args.retention<1:raise ValueError('--retention must be between 0 and 1')
    if args.epochs<=args.mask_warmup: raise ValueError('--epochs must exceed --mask-warmup')
    stage_models={1:['A0','A1','A2'],2:['A0','A1','A2','A3'],3:['A0','A1','A2','A3','A4'],4:['A0','A1','A2','A3','A4','A5']}
    if args.models is None:args.models=stage_models[args.stage]
    if any(x not in stage_models[args.stage] for x in args.models):raise ValueError('Model does not belong to selected stage')
    if args.output is None:args.output=f'results/stage{args.stage}/{args.dataset}'
    Path(args.output).mkdir(parents=True,exist_ok=True)
    result_file=Path(args.output)/'results.csv'
    if result_file.exists(): raise FileExistsError(f'{result_file} already exists; choose a new output directory')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if min(args.lambda_fid,args.lambda_budget,args.lambda_dp,args.lambda_eo,args.lambda_mask_fair,args.lambda_robust,args.lambda_mask_stability,args.lambda_distill) < 0: raise ValueError('Loss weights must be nonnegative')
    if args.attack_budget < 0 or args.drift_steps < 1: raise ValueError('Invalid attack budget or steps')
    for seed in args.seeds:
        seed_rows=[]
        # Optional split-per-seed NPZ takes priority; otherwise fixed split for all seeds.
        specific=Path(args.data_dir)/f'{args.dataset}_seed{seed}.npz'
        path=specific if specific.exists() else Path(args.data_dir)/f'{args.dataset}.npz'
        if not path.exists():raise FileNotFoundError(f'Missing {path}; supply raw graph data and original splits')
        data=load_npz(path,device)
        for name in stage_models[args.stage]:
            if name not in args.models:continue
            row=train_one(data,name,seed,args,device)
            seed_rows.append(row)
            outfile=result_file; exists=outfile.exists()
            with outfile.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row));
                if not exists:w.writeheader()
                w.writerow(row)
            print(json.dumps(row,allow_nan=True))
        decision=choose_checkpoint(seed_rows,GateConfig(
            max_dp=args.max_dp,max_eo=args.max_eo,max_auc_drop=args.max_auc_drop,
            max_prediction_drift=args.max_prediction_drift))
        decision.update(dataset=args.dataset,seed=seed)
        (Path(args.output)/f'fallback_seed{seed}.json').write_text(json.dumps(decision,indent=2,allow_nan=False))
        print('GLOBAL FALLBACK:',json.dumps(decision))

if __name__=='__main__':main()
