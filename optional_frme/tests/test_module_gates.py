from frme.controller import explanation_fair_stable_gate, graph_route_fair_stable_gate

def test_graph_joint_accept():
    g={'auc':.8,'dp_gap':.03,'eo_gap':.04}
    f={'auc':.7,'dp_gap':.04,'eo_gap':.04}
    stability={'preservation':.99,'probability_drift':.01}
    assert graph_route_fair_stable_gate(g,f,stability,stability).passed

def test_graph_joint_reject_fairness():
    g={'auc':.8,'dp_gap':.3,'eo_gap':.04}
    f={'auc':.7,'dp_gap':.04,'eo_gap':.04}
    stability={'preservation':.99,'probability_drift':.01}
    assert graph_route_fair_stable_gate(g,f,stability,stability).route=='features'

def test_mask_fairness_stability_gate():
    valid={'scores':{'std':.1},'order_invariant':True,
           'learned':{'sufficiency_mae':.01,'group_sufficiency_gap':.02},
           'random':{'sufficiency_mae':.08},'stable_common_edge_jaccard':.7,
           'prediction_preservation_rate':.99}
    assert explanation_fair_stable_gate(valid).passed
    valid['learned']['group_sufficiency_gap']=.2
    assert explanation_fair_stable_gate(valid).route=='abstain'
    valid['learned']['group_sufficiency_gap']=.02
    valid['stable_common_edge_jaccard']=None
    assert explanation_fair_stable_gate(valid).route=='abstain'
