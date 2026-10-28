import torch
from frme.core import synthetic_graph, GCN
from frme.graph_bridge import FairStableGraphBridge, train_fair_stable_bridge
from frme.train import train_explainer
from frme.retention import freeze_teacher,replay_memory

def test_fair_stable_graph_bridge_trains_and_has_gradients():
    d=synthetic_graph(n=60,seed=8)
    model=GCN(d['x'].shape[1])
    b=train_fair_stable_bridge(model,d,epochs=2,seed=9)
    weights=b(d['x'],d['edges'])
    assert weights.shape==(d['edges'].shape[1],)
    assert torch.isfinite(weights).all()
    assert ((weights>=0)&(weights<=1)).all()
    weights.mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())

def test_fair_explanation_training_runs():
    d=synthetic_graph(n=50,seed=8)
    model=GCN(d['x'].shape[1])
    expl=train_explainer(model,d,epochs=2,lam_explain_fair=.25,lam_rob=.1)
    with torch.no_grad():
        scores=expl(model.embed(d['x'],d['edges']),d['edges'])
    assert torch.isfinite(scores).all()

def test_retention_independent_of_federated_code():
    d=synthetic_graph(n=50,seed=3)
    model=GCN(d['x'].shape[1]);teacher=freeze_teacher(model);memory=replay_memory(d,cap=12)
    assert len(memory['idx'])==12
    assert not any(p.requires_grad for p in teacher.parameters())
