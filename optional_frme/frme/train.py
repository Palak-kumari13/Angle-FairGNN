import copy
import torch
from torch.nn import functional as F
from .core import (GCN,MLP,EdgeExplainer,parity_gap,opportunity_gap,kd,topk_mask,matched_random_mask,degree_mask,perturb_edges,mask_score_diagnostics,check_mask_permutation,metric)


def train_predictor(data,epochs=100,seed=0,teacher=None,replay=None,weights=None,lr=.01,lam_fair=0.,lam_kd=0.,lam_replay=0.,lam_prox=0.,prev=None,init_model=None,model_type="gcn"):
    torch.manual_seed(seed);x,e,y,s=data['x'],data['edges'],data['y'],data['s'];tr=data['train'];val=data['val']
    m=copy.deepcopy(init_model) if init_model is not None else (MLP(x.shape[1]) if model_type=="mlp" else GCN(x.shape[1]));
    for parameter in m.parameters():parameter.requires_grad_(True)
    opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-4)
    best=float('inf');best_state=None
    for epoch in range(epochs):
        m.train();opt.zero_grad();logits=m(x,e);p=torch.sigmoid(logits[tr]);loss=F.binary_cross_entropy_with_logits(logits[tr],y[tr]);loss=loss+lam_fair*(parity_gap(p,s[tr])+opportunity_gap(p,y[tr],s[tr]))
        if teacher is not None:
            with torch.no_grad(): tl=teacher(x,e)
            loss+=lam_kd*kd(logits[tr],tl[tr])
        if replay is not None and teacher is not None:
            # Rehearsal on stored historical graph/labels; not current-graph indices.
            xr,er,yr,ir=replay['x'],replay['edges'],replay['y'],replay['idx']
            loss+=lam_replay*F.binary_cross_entropy_with_logits(m(xr,er)[ir],yr[ir])
        if prev is not None and lam_prox:
            loss+=lam_prox*sum((a-b.detach()).square().sum() for a,b in zip(m.parameters(),prev.parameters()))
        loss.backward();opt.step()
        m.eval()
        with torch.no_grad():v=F.binary_cross_entropy_with_logits(m(x,e)[val],y[val]).item()
        if v<best:best=v;best_state=copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state);return m


def train_explainer(model,data,epochs=100,seed=0,budget=.5,lr=.005,lam_rob=0.,lam_entropy=.01,lam_explain_fair=0.):
    """Freeze teacher; optimize masked sufficiency + differentiable deletion/comprehensiveness.
    Train using TRAIN nodes only. Mask is global over the graph; evaluation is transductive.
    """
    torch.manual_seed(seed);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    x,e=data['x'],data['edges'];tr=data['train'];ex=EdgeExplainer(model.l1.out_features)
    opt=torch.optim.Adam(ex.parameters(),lr=lr)
    with torch.no_grad():z=model.embed(x,e).detach();orig=model(x,e).detach()
    for t in range(epochs):
        ex.train();opt.zero_grad();scores=ex(z,e);probs=torch.sigmoid(scores)
        # A soft retention budget avoids globally saturated all-one masks.
        retained=model(x,e,probs);deleted=model(x,e,1-probs)
        fidelity=kd(retained[tr],orig[tr]);deletion=F.relu(.15-(torch.sigmoid(orig[tr])-torch.sigmoid(deleted[tr])).abs().mean())
        budget_loss=(probs.mean()-budget).square()
        entropy=(-probs.clamp(1e-6,1-1e-6)*probs.clamp(1e-6,1-1e-6).log()-(1-probs).clamp(1e-6,1-1e-6)*(1-probs).clamp(1e-6,1-1e-6).log()).mean()
        loss=fidelity+0.2*deletion+4*budget_loss+lam_entropy*entropy
        if lam_explain_fair:
            # Fairness proxy: parity of retention-induced prediction error on TRAIN nodes.
            group=data['s'][tr]
            err=(torch.sigmoid(retained[tr])-torch.sigmoid(orig[tr])).abs()
            if (group==0).any() and (group==1).any():
                loss+=lam_explain_fair*(err[group==0].mean()-err[group==1].mean()).abs()
        if lam_rob:
            ep=perturb_edges(e,x.shape[0],.05)
            with torch.no_grad(): pert_logits=model(x,ep);zp=model.embed(x,ep)
            unchanged=(torch.sigmoid(pert_logits[tr])>.5)==(torch.sigmoid(orig[tr])>.5)
            # Only stabilize scores of edges unchanged by sampled perturbation.
            common=(e==ep).all(dim=0)
            if unchanged.float().mean()>.95 and common.any():
                ps=ex(zp,ep);loss+=lam_rob*F.mse_loss(scores[common].sigmoid(),ps[common].sigmoid())
        loss.backward();opt.step()
    return ex


def evaluate_explainer(model,ex,data,budget=.5,seed=0,split="test"):
    if split not in {"val", "test"}: raise ValueError("split must be val or test")
    torch.manual_seed(seed);model.eval();ex.eval();x,e=data['x'],data['edges'];idx=data[split]
    with torch.no_grad():
        z=model.embed(x,e);scores=ex(z,e);ref=torch.sigmoid(model(x,e)[idx]);d={}
        for name,mask in [('learned',topk_mask(scores,budget)),('random',matched_random_mask(scores,budget)),('degree',degree_mask(e,x.shape[0],budget))]:
            retain=torch.sigmoid(model(x,e,mask)[idx]);remove=torch.sigmoid(model(x,e,1-mask)[idx]);d[name]={'sufficiency_mae':float((ref-retain).abs().mean()),'comprehensiveness_mae':float((ref-remove).abs().mean()),'retained_edges':int(mask.sum())}
            # Group-wise faithfulness disparity (diagnostic fairness proxy, not causal fairness).
            grp=data['s'][idx]
            if (grp==0).any() and (grp==1).any():
                err=(ref-retain).abs()
                d[name]['group_sufficiency_gap']=float((err[grp==0].mean()-err[grp==1].mean()).abs())
            else:d[name]['group_sufficiency_gap']=None
        d['scores']=mask_score_diagnostics(scores)
        d['order_invariant']=check_mask_permutation(ex,z,e)
        d['prediction']=metric(model(x,e),data['y'],data['s'],idx)
        ep=perturb_edges(e,x.shape[0],.05)
        changed=(e!=ep).any(dim=0);common=~changed
        new_pred=torch.sigmoid(model(x,ep)[idx]);keep=(ref>.5)==(new_pred>.5)
        if common.any() and keep.all():
            zp=model.embed(x,ep);ps=ex(zp,ep)
            # top-k overlap restricted to unchanged edges; excludes artificially inserted/deleted edges.
            a=topk_mask(scores[common],budget).bool();b=topk_mask(ps[common],budget).bool()
            d['stable_common_edge_jaccard']=float((a&b).sum()/(a|b).sum().clamp_min(1))
        else:d['stable_common_edge_jaccard']=None
        d['prediction_preservation_rate']=float(keep.float().mean())
    return d
