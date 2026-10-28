"""Graph construction / structural validation / deterministic module fallback.
Research scaffold: valid topology != predictive graph signal. Never infer graph quality from density alone.
"""
from dataclasses import dataclass, asdict
import torch

@dataclass
class GraphGate:
    accepted: bool
    reason: str
    nodes: int
    edges: int
    isolated_fraction: float
    components: int
    self_loops: int
    duplicate_edges: int


def graph_audit(edges, n, *, max_isolated=.35, max_components_fraction=.25, min_edges=1):
    if n < 2:
        return GraphGate(False,'too_few_nodes',n,0,1.,n,0,0)
    if edges is None or edges.numel() == 0:
        return GraphGate(False,'empty_graph',n,0,1.,n,0,0)
    if edges.ndim!=2 or edges.shape[0]!=2:
        return GraphGate(False,'invalid_edge_shape',n,0,1.,n,0,0)
    e=edges.detach().cpu().long()
    if (e<0).any() or (e>=n).any():
        return GraphGate(False,'invalid_endpoint',n,e.shape[1],1.,n,0,0)
    ids=e[0]*n+e[1]
    dup=int(e.shape[1]-torch.unique(ids).numel()); loops=int((e[0]==e[1]).sum())
    # Undirected connectivity: directed edges still count as adjacency for components.
    parent=list(range(n))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    for a,b in e.T.tolist():
        ra,rb=find(a),find(b)
        if ra!=rb:parent[ra]=rb
    components=len({find(i) for i in range(n)})
    touched=torch.zeros(n,dtype=torch.bool)
    nonloop=e[:,e[0]!=e[1]]
    if nonloop.numel():touched[nonloop.flatten()]=True
    iso=float((~touched).float().mean())
    reasons=[]
    if e.shape[1]<min_edges:reasons.append('too_few_edges')
    if iso>max_isolated:reasons.append('high_isolation')
    if components/n>max_components_fraction:reasons.append('too_many_components')
    return GraphGate(not reasons,','.join(reasons) if reasons else 'structurally_valid',n,e.shape[1],iso,components,loops,dup)


def knn_edges(x,k=8,*,max_nodes=5000,mutual=False):
    """Feature cosine kNN, excluding self; capped as dense O(N²) prototype."""
    import torch.nn.functional as F
    n=x.shape[0]
    if n<2:raise ValueError('need >= 2 nodes')
    if n>max_nodes:raise ValueError('dense kNN limit exceeded; use approximate neighbor index')
    if not torch.isfinite(x).all():raise ValueError('features contain NaN/Inf')
    z=F.normalize(x.float(),dim=-1,eps=1e-8)
    similarity=z@z.T
    similarity.fill_diagonal_(-torch.inf)
    idx=similarity.topk(min(k,n-1),dim=1).indices
    src=torch.arange(n,device=x.device).repeat_interleave(idx.shape[1]);dst=idx.flatten()
    edge=torch.stack([src,dst])
    if mutual:
        ids=src*n+dst;rev=dst*n+src
        keep=torch.isin(ids,rev)
        edge=edge[:,keep]
    # Bidirectional for symmetric message flow; make the construction deterministic.
    edge=torch.cat([edge,edge.flip(0)],dim=1)
    edge=torch.unique(edge.T,dim=0).T.contiguous()
    return edge


def bridge_graph(x, original_edges=None, *, k=8, max_isolated=.35, max_components_fraction=.25,
                 mode='auto', max_nodes=5000):
    """Modes auto/original/knn/features. No silent assertion of useful graph signal.
    Ordered fallbacks: valid original -> constructed kNN -> feature-only.
    """
    if mode not in {'auto','original','knn','features'}:raise ValueError('unknown mode')
    n=x.shape[0];attempts=[]
    if mode=='features':return {'edges':None,'route':'features','accepted':True,'attempts':[],'requires_signal_test':False}
    if mode in {'auto','original'}:
        gate=graph_audit(original_edges,n,max_isolated=max_isolated,max_components_fraction=max_components_fraction)
        attempts.append({'route':'original','gate':asdict(gate)})
        if gate.accepted:return {'edges':original_edges,'route':'original','accepted':True,'attempts':attempts,'requires_signal_test':True}
    if mode in {'auto','knn'}:
        try:
            proposed=knn_edges(x,k,max_nodes=max_nodes)
            gate=graph_audit(proposed,n,max_isolated=max_isolated,max_components_fraction=max_components_fraction)
            attempts.append({'route':'knn','gate':asdict(gate)})
            if gate.accepted:return {'edges':proposed,'route':'knn','accepted':True,'attempts':attempts,'requires_signal_test':True}
        except (ValueError,RuntimeError) as ex:
            attempts.append({'route':'knn','error':str(ex)})
    return {'edges':None,'route':'features','accepted':True,'attempts':attempts,'requires_signal_test':False}


def validation_fallback(graph_metric, feature_metric, *, minimum_gain=0.0):
    """Use *validation* AUROC (or same prespecified metric), never held-out test labels.
    Fail-closed on NaN, missing metrics, or insufficient observed graph benefit.
    """
    import math
    if graph_metric is None or feature_metric is None or not all(math.isfinite(v) for v in (graph_metric,feature_metric)):
        return 'features','missing_or_invalid_validation_metric'
    if graph_metric < feature_metric+minimum_gain:
        return 'features','graph_has_no_validation_gain'
    return 'graph','graph_validated'


class FairStableGraphBridge(torch.nn.Module):
    """Trainable edge-weight reconstruction on a FIXED candidate topology.

    Learns soft edge weights; it cannot invent edges or certify group fairness.
    Train only against train nodes, and gate on validation nodes.
    """
    def __init__(self,d_in,hidden=32):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(3*d_in,hidden),torch.nn.ReLU(),torch.nn.Linear(hidden,1))
    def forward(self,x,edges):
        src,dst=edges
        features=torch.cat((x[src],x[dst],(x[src]-x[dst]).abs()),dim=-1)
        return self.net(features).flatten().sigmoid()

def train_fair_stable_bridge(model,data,epochs=30,seed=0,lr=.005,
                             lam_fair=.2,lam_stability=.2,lam_budget=.1,
                             lam_distill=.1,budget=.5,edit_rate=.05):
    """Learn fair and perturbation-stable edge weights, against a frozen trained predictor.
    Structural edits use index-aligned replacement; this is a stability proxy.
    """
    from torch.nn import functional as F
    from .core import parity_gap,opportunity_gap,perturb_edges,kd
    torch.manual_seed(seed)
    x,e,y,s=data['x'],data['edges'],data['y'],data['s'];tr=data['train']
    if e is None or e.numel()==0:raise ValueError('cannot reconstruct from an empty candidate topology')
    bridge=FairStableGraphBridge(x.shape[-1]).to(x.device)
    teacher=__import__('copy').deepcopy(model).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    opt=torch.optim.Adam(bridge.parameters(),lr=lr)
    with torch.no_grad():reference=teacher(x,e)
    for step in range(epochs):
        opt.zero_grad();weights=bridge(x,e)
        logits=teacher(x,e,weights)
        prob=logits[tr].sigmoid()
        loss=F.binary_cross_entropy_with_logits(logits[tr],y[tr])
        loss=loss+lam_fair*(parity_gap(prob,s[tr])+opportunity_gap(prob,y[tr],s[tr]))
        loss=loss+lam_budget*(weights.mean()-budget).square()
        loss=loss+lam_distill*kd(logits[tr],reference[tr])
        if lam_stability:
            ep=perturb_edges(e,x.shape[0],edit_rate)
            wp=bridge(x,ep)
            altered=teacher(x,ep,wp)
            common=(e==ep).all(dim=0)
            structural=F.mse_loss(weights[common],wp[common]) if common.any() else weights.new_zeros(())
            loss=loss+lam_stability*(F.mse_loss(logits[tr].sigmoid(),altered[tr].sigmoid())+structural)
        loss.backward();opt.step()
    return bridge
