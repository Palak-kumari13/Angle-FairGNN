"""End-to-end research scaffold: graph bridge + joint fairness/stability gates + mask abstention + retention gate.
Synthetic smoke only; no real-dataset performance claims.
"""
import argparse, json, os
from frme.core import synthetic_graph,seed_all,metric
from frme.graph_bridge import bridge_graph,train_fair_stable_bridge
from frme.train import train_predictor, train_explainer, evaluate_explainer
from frme.controller import (evaluate_module_stack, explanation_fair_stable_gate, retention_gate,
                             graph_route_fair_stable_gate, prediction_stability)
from frme.retention import freeze_teacher,replay_memory

p=argparse.ArgumentParser()
p.add_argument('--epochs',type=int,default=35)
p.add_argument('--seeds',type=int,default=1)
p.add_argument('--out',default='results/integrated.json')
p.add_argument('--budget',type=float,default=.5)
p.add_argument('--max-dp',type=float,default=.10)
p.add_argument('--max-eo',type=float,default=.10)
p.add_argument('--max-auc-drop',type=float,default=.03)
p.add_argument('--min-preservation',type=float,default=.90)
p.add_argument('--max-drift',type=float,default=.10)
p.add_argument('--train-bridge',action='store_true')
p.add_argument('--explanation-fair-weight',type=float,default=.3)
a=p.parse_args()
os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
results=[]
for seed in range(a.seeds):
    seed_all(seed);d=synthetic_graph(seed=seed)
    bridge=bridge_graph(d['x'],d['edges'])
    if bridge['edges'] is not None:d['edges']=bridge['edges']
    feature=train_predictor(d,epochs=a.epochs,seed=seed,model_type='mlp')
    selected=feature;route='features';joint=None;graph_gate=None;bridge_learning=None;explanation={'status':'abstain_no_graph'}
    if bridge['edges'] is not None:
        graph=train_predictor(d,epochs=a.epochs,seed=seed,model_type='gcn')
        gm=metric(graph(d['x'],d['edges']),d['y'],d['s'],d['val'])
        fm=metric(feature(d['x']),d['y'],d['s'],d['val'])
        gs=prediction_stability(graph,d,seed=seed)
        fs=prediction_stability(feature,d,seed=seed)
        graph_gate=graph_route_fair_stable_gate(gm,fm,gs,fs,
                     min_preservation=a.min_preservation,max_drift=a.max_drift)
        route=graph_gate.route
        if route=='graph':
            if a.train_bridge:
                learned_bridge=train_fair_stable_bridge(graph,d,epochs=a.epochs,seed=seed)
                # Bridge weights are learned, evaluated without test-driven selection.
                # The current weighted-edge predictor is separately evaluated, not silently substituted.
                from frme.core import parity_gap, opportunity_gap
                with __import__('torch').no_grad():
                    weights=learned_bridge(d['x'],d['edges'])
                    bridge_learning={'mean_weight':float(weights.mean()),'std_weight':float(weights.std(unbiased=False)),
                                     'note':'Experimental weighted topology; trained with fairness and stability losses. Still requires joint gate and ablations.'}
            # Baseline is a reference checkpoint, not a certified fair/robust model.
            candidate=train_predictor(d,epochs=a.epochs,seed=seed,model_type='gcn',
                                      init_model=graph,lam_fair=.15,lr=.005)
            selected,joint=evaluate_module_stack(graph,candidate,d,seed=seed,
                       max_dp=a.max_dp,max_eo=a.max_eo,max_auc_drop=a.max_auc_drop,
                       min_preservation=a.min_preservation,max_drift=a.max_drift)
            ex=train_explainer(selected,d,epochs=a.epochs,seed=seed,budget=a.budget,lam_rob=.2,lam_explain_fair=a.explanation_fair_weight)
            # Gate selection uses validation only; test diagnostics are reporting-only.
            er_val=evaluate_explainer(selected,ex,d,budget=a.budget,seed=seed,split='val')
            eg=explanation_fair_stable_gate(er_val)
            er_test=evaluate_explainer(selected,ex,d,budget=a.budget,seed=seed,split='test')
            explanation={'status':eg.route,'gate':eg.report(),
                         'validation_diagnostics':er_val,'test_diagnostics':er_test}
    old=freeze_teacher(selected)
    new=synthetic_graph(seed=seed+900)
    updated=train_predictor(new,epochs=a.epochs,seed=seed,init_model=old,
                            teacher=old,replay=replay_memory(d,seed=seed),
                            lam_fair=.05,lam_replay=.4,lam_kd=.1,lam_prox=.001,prev=old)
    old_before=metric(selected(d['x'],d['edges']),d['y'],d['s'],d['val'])
    old_after=metric(updated(d['x'],d['edges']),d['y'],d['s'],d['val'])
    retain=retention_gate(old_before,old_after)
    active=updated if retain.passed else old
    with __import__('torch').no_grad():
        test_metrics=metric(active(d['x'],d['edges']),d['y'],d['s'],d['test'])
    results.append({'seed':seed,'bridge':{'route':bridge['route'],'attempts':bridge['attempts'],
                    'graph_utility_route':route, 'joint_graph_gate':graph_gate.report() if graph_gate else None,'bridge_learning':bridge_learning},'joint_validation':joint,
                    'explanation':explanation,'retention':retain.report(),'test_metrics':test_metrics,
                    'warning':'Synthetic demonstration; no method outperformance established.'})
with open(a.out,'w') as f:json.dump(results,f,indent=2)
print('Saved:',a.out)
for row in results:
    print('seed',row['seed'],'graph',row['bridge']['graph_utility_route'],
          'joint',row['joint_validation']['selected'] if row['joint_validation'] else 'N/A',
          'mask',row['explanation']['status'],'retention',row['retention']['route'])
