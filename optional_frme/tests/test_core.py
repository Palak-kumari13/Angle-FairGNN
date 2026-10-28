import torch
from frme.core import synthetic_graph,GCN,EdgeExplainer,topk_mask,aggregate,check_mask_permutation,parity_gap
from frme.retention import freeze_teacher,replay_memory

def test_mask_budget():
    s=torch.tensor([.1,.4,.3,.2]);assert topk_mask(s,.5).tolist()==[0.,1.,1.,0.]
def test_shapes_and_grad():
    d=synthetic_graph(n=40);m=GCN();x=m(d['x'],d['edges']);assert x.shape==(40,)
    z=m.embed(d['x'],d['edges']);ex=EdgeExplainer();sc=ex(z,d['edges']);sc.mean().backward();assert any(p.grad is not None for p in ex.parameters())
def test_edge_order_equivariance():
    d=synthetic_graph(n=40);m=GCN();ex=EdgeExplainer();assert check_mask_permutation(ex,m.embed(d['x'],d['edges']),d['edges'])
def test_replay_teacher():
    d=synthetic_graph(n=40);m=GCN();t=freeze_teacher(m);assert not any(p.requires_grad for p in t.parameters());r=replay_memory(d,8);assert len(r['idx'])==8
def test_fairness_zero_groups():
    assert parity_gap(torch.rand(5),torch.zeros(5,dtype=torch.long)).item()==0

from frme.graph_bridge import graph_audit,knn_edges,bridge_graph,validation_fallback
from frme.core import MLP

def test_graph_bridge_empty_graph_falls_back_knn():
    x=torch.randn(24,5)
    result=bridge_graph(x,torch.empty((2,0),dtype=torch.long),k=4)
    assert result['route']=='knn' and result['edges'].shape[0]==2
    assert result['attempts'][0]['gate']['reason']=='empty_graph'

def test_graph_bridge_invalid_features_fallback():
    x=torch.zeros(5,3);x[0,0]=float('nan')
    result=bridge_graph(x,None,k=2)
    assert result['route']=='features' and any('error' in r for r in result['attempts'])

def test_graph_bridge_bad_endpoints_and_isolation():
    assert graph_audit(torch.tensor([[0],[99]]),5).reason=='invalid_endpoint'
    g=graph_audit(torch.tensor([[0,1],[1,0]]),30)
    assert not g.accepted and g.isolated_fraction>.9

def test_validation_fallback():
    assert validation_fallback(.70,.80)[0]=='features'
    assert validation_fallback(.86,.80,minimum_gain=.02)[0]=='graph'
    assert validation_fallback(None,.8)[0]=='features'

def test_feature_model_works_without_graph():
    x=torch.randn(20,8);assert MLP()(x,None).shape==(20,)

from frme.controller import fairness_gate, stability_gate, explanation_gate, retention_gate, evaluate_module_stack

def test_fail_closed_fairness():
    g=fairness_gate({'auc':.8,'dp_gap':None,'eo_gap':.04},{'auc':.81})
    assert not g.passed and g.route=='reference'

def test_fairness_utility_tradeoff():
    assert not fairness_gate({'auc':.60,'dp_gap':0.,'eo_gap':0.},{'auc':.85}).passed

def test_stability_rejects_drift():
    assert not stability_gate({'preservation':.99,'probability_drift':.2},{'probability_drift':.01}).passed

def test_mask_abstention_on_equal_random():
    g=explanation_gate({'scores':{'std':.02},'order_invariant':True,'learned':{'sufficiency_mae':.11},'random':{'sufficiency_mae':.11}})
    assert g.route=='abstain'

def test_retention_rollback():
    assert retention_gate({'auc':.9},{'auc':.75}).route=='previous_checkpoint'

def test_joint_controller_runs():
    from frme.core import synthetic_graph,MLP
    d=synthetic_graph(n=80,seed=4);m=MLP(d['x'].size(1))
    _,report=evaluate_module_stack(m,m,d,min_preservation=0.,max_drift=1.)
    assert 'fairness' in report and 'robustness' in report
