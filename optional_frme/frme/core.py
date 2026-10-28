"""Small dependency-light graph baseline; research prototype, not FairGNN/PGExplainer reimplementation."""
import copy
import torch
from torch import nn
import torch.nn.functional as F


def seed_all(seed):
    torch.manual_seed(seed)


def aggregate(x, edges, weights=None):
    """Directed weighted message sum with symmetric degree normalization; self-loops added."""
    n=x.size(0); src,dst=edges
    w=torch.ones(src.numel(),device=x.device,dtype=x.dtype) if weights is None else weights
    ids=torch.arange(n,device=x.device)
    src=torch.cat([src,ids]);dst=torch.cat([dst,ids]);w=torch.cat([w,torch.ones(n,device=x.device,dtype=x.dtype)])
    deg=torch.zeros(n,device=x.device).index_add(0,dst,w)
    out=torch.zeros_like(x).index_add(0,dst,x[src]*(w/(deg[src].clamp_min(1e-6)*deg[dst].clamp_min(1e-6)).sqrt())[:,None])
    return out


class GCN(nn.Module):
    def __init__(self,d_in=8,hidden=32):
        super().__init__(); self.l1=nn.Linear(d_in,hidden);self.l2=nn.Linear(hidden,1)
    def embed(self,x,edges,weights=None):
        return F.relu(self.l1(aggregate(x,edges,weights)))
    def forward(self,x,edges,weights=None):
        return self.l2(aggregate(self.embed(x,edges,weights),edges,weights)).flatten()


class MLP(nn.Module):
    """Feature-only fallback. Signature compatible with GCN for pipeline reuse."""
    def __init__(self,d_in=8,hidden=32):
        super().__init__();self.l1=nn.Linear(d_in,hidden);self.l2=nn.Linear(hidden,1)
    def embed(self,x,edges=None,weights=None):
        return F.relu(self.l1(x))
    def forward(self,x,edges=None,weights=None):
        return self.l2(self.embed(x)).flatten()


class EdgeExplainer(nn.Module):
    """PGExplainer-inspired inductive scorer, with differentiable soft masking."""
    def __init__(self,d_hidden=32):
        super().__init__();self.net=nn.Sequential(nn.Linear(4*d_hidden,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,z,edges):
        a,b=z[edges[0]],z[edges[1]]
        return self.net(torch.cat([a,b,(a-b).abs(),a*b],-1)).flatten()


def topk_mask(scores,k_fraction=0.5):
    k=max(1,round(scores.numel()*k_fraction));idx=torch.argsort(scores,descending=True,stable=True)[:k]
    m=torch.zeros_like(scores);m[idx]=1.0
    return m


def parity_gap(p,s):
    if (s==0).sum()==0 or (s==1).sum()==0:return p.new_zeros(())
    return (p[s==0].mean()-p[s==1].mean()).abs()


def opportunity_gap(p,y,s):
    a=(s==0)&(y==1);b=(s==1)&(y==1)
    if not a.any() or not b.any():return p.new_zeros(())
    return (p[a].mean()-p[b].mean()).abs()


def kd(student_logits, teacher_logits,temperature=2.0):
    # Binary Bernoulli distillation, teacher detached.
    target=torch.sigmoid(teacher_logits.detach()/temperature)
    return F.binary_cross_entropy_with_logits(student_logits/temperature,target)*temperature**2


def matched_random_mask(scores,fraction,generator=None):
    k=max(1,round(scores.numel()*fraction));perm=torch.randperm(scores.numel(),device=scores.device,generator=generator)
    mask=torch.zeros_like(scores);mask[perm[:k]]=1
    return mask


def degree_mask(edges,n,fraction):
    deg=torch.bincount(edges.flatten(),minlength=n).float().to(edges.device)
    sc=deg[edges[0]]+deg[edges[1]]
    return topk_mask(sc,fraction)


def consistency_kl(a,b):
    return kd(b,a,1.0)


def mask_score_diagnostics(scores):
    s=scores.detach();return {'std':float(s.std(unbiased=False)),'span':float(s.max()-s.min()),'near_constant':bool(s.std(unbiased=False)<1e-4)}


def check_mask_permutation(explainer,z,edges,atol=1e-5):
    with torch.no_grad():
        original=explainer(z,edges);p=torch.randperm(edges.shape[1],device=edges.device)
        return bool(torch.allclose(original[p],explainer(z,edges[:,p]),atol=atol))


def perturb_edges(edges,n,rate=0.05):
    # Edge order preserved for surviving edges; changed edges replaced (no duplicated checks).
    altered=edges.clone();num=max(1,int(edges.shape[1]*rate));p=torch.randperm(edges.shape[1],device=edges.device)[:num]
    altered[1,p]=torch.randint(n,(num,),device=edges.device)
    return altered


def synthetic_graph(n=160,d=8,seed=0):
    """Toy graph with feature + homophily signal. Not a ground-truth edge-importance dataset."""
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(n,d,generator=g);s=(x[:,1]>0).long()
    y=((x[:,0]+0.4*x[:,2])>0).float()
    e=[]
    for i in range(n):
        for j in torch.randperm(n,generator=g)[:8].tolist():
            if i!=j and (y[i]==y[j] or torch.rand((),generator=g).item()<.15):e.append((i,j))
    edges=torch.tensor(e).T.contiguous();p=torch.randperm(n,generator=g)
    return {'x':x,'edges':edges,'y':y,'s':s,'train':p[:int(.6*n)],'val':p[int(.6*n):int(.8*n)],'test':p[int(.8*n):]}


def metric(logits,y,s,idx):
    from sklearn.metrics import roc_auc_score,f1_score,balanced_accuracy_score,average_precision_score,matthews_corrcoef
    p=torch.sigmoid(logits[idx]).detach().cpu().numpy();yt=y[idx].detach().cpu().numpy().astype(int)
    pred=(p>=.5).astype(int);ss=s[idx].detach().cpu().numpy()
    out={'f1':float(f1_score(yt,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(yt,pred)),'mcc':float(matthews_corrcoef(yt,pred)),'dp_gap':abs(float(pred[ss==0].mean()-pred[ss==1].mean())) if (ss==0).any() and (ss==1).any() else None}
    out['auc']=float(roc_auc_score(yt,p)) if len(set(yt))>1 else None
    out['auprc']=float(average_precision_score(yt,p)) if len(set(yt))>1 else None
    pos0=(ss==0)&(yt==1);pos1=(ss==1)&(yt==1)
    out['eo_gap']=abs(float(pred[pos0].mean()-pred[pos1].mean())) if pos0.any() and pos1.any() else None
    return out
