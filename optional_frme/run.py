import argparse,json,os
from frme.core import synthetic_graph,seed_all,metric
from frme.graph_bridge import bridge_graph,validation_fallback
from frme.train import train_predictor,train_explainer,evaluate_explainer
from frme.retention import freeze_teacher,replay_memory

p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=2);p.add_argument('--epochs',type=int,default=60);p.add_argument('--out',default='results/demo.json');p.add_argument('--budget',type=float,default=.5);p.add_argument('--robust',action='store_true');p.add_argument('--graph-mode',choices=['auto','original','knn','features'],default='auto');p.add_argument('--min-graph-gain',type=float,default=0.0);a=p.parse_args()
os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True);results=[]
for seed in range(a.seeds):
    seed_all(seed);d=synthetic_graph(seed=seed)
    routing=bridge_graph(d['x'],d['edges'],mode=a.graph_mode)
    if routing['edges'] is not None:d['edges']=routing['edges']
    # Validate graph utility only on held-out validation nodes (never test data).
    feature_model=train_predictor(d,epochs=a.epochs,seed=seed,model_type='mlp')
    if routing['edges'] is not None:
        graph_model=train_predictor(d,epochs=a.epochs,seed=seed,model_type='gcn')
        vg=metric(graph_model(d['x'],d['edges']),d['y'],d['s'],d['val'])['auc']
        vf=metric(feature_model(d['x']),d['y'],d['s'],d['val'])['auc']
        selected,reason=validation_fallback(vg,vf,minimum_gain=a.min_graph_gain)
    else:
        selected,reason='features','no_structurally_valid_graph';vg=None;vf=metric(feature_model(d['x']),d['y'],d['s'],d['val'])['auc']
    base=graph_model if selected=='graph' else feature_model
    # Edge explanations are explicitly withheld if graph predictive signal is not validated.
    if selected=='graph':
        expl=train_explainer(base,d,epochs=a.epochs,seed=seed,budget=a.budget,lam_rob=.2 if a.robust else 0)
        explanation=evaluate_explainer(base,expl,d,budget=a.budget,seed=seed)
        sd=explanation['scores'];explanation['status']='candidate' if not sd['near_constant'] and explanation['order_invariant'] else 'failed_mask_integrity'
    else:
        explanation={'status':'not_applicable_feature_fallback','learned':None}
    old=freeze_teacher(base)
    new=synthetic_graph(seed=seed+900)
    # Route new task with the same encoder family; MLP ignores edge argument.
    updated=train_predictor(new,epochs=a.epochs,seed=seed,teacher=old,replay=replay_memory(d,seed=seed),lam_fair=.05,lam_replay=.4,lam_prox=.001,prev=old,init_model=old)
    old_before=metric(base(d['x'],d['edges'] if selected=='graph' else None),d['y'],d['s'],d['test'])
    old_after=metric(updated(d['x'],d['edges'] if selected=='graph' else None),d['y'],d['s'],d['test'])
    results.append({'seed':seed,'graph_bridge':{'route':routing['route'],'attempts':routing['attempts'],'chosen':selected,'reason':reason,'validation_graph_auc':vg,'validation_feature_auc':vf},'explanation':explanation,'previous_task_before':old_before,'previous_task_after':old_after,'old_task_auc_forgetting':old_before['auc']-old_after['auc'] if old_before['auc'] is not None and old_after['auc'] is not None else None})
with open(a.out,'w') as f:json.dump(results,f,indent=2)
print('Saved',a.out)
for r in results:print('seed',r['seed'],'score_std',round(r['explanation'].get('scores',{}).get('std',0),6),'mask_status',r['explanation']['status'],'old_auc_forgetting',r['old_task_auc_forgetting'])
