"""Undirected edge edits, prediction stability, and explanation drift.
All scores use the same fixed evaluation perturbations across models and seeds.
"""
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score


def edit_edges(edges, n, budget, seed):
    """Flip up to budget undirected non-self edges; deterministic for (seed, graph)."""
    pairs = {tuple(p) for p in edges.T.detach().cpu().tolist()}
    if budget == 0:
        return edges
    rng = np.random.default_rng(seed)
    # Half deletions and half additions, as available.
    old = sorted(pairs)
    deletes = min(len(old), budget // 2)
    if deletes:
        for ix in rng.choice(len(old), deletes, replace=False):
            pairs.remove(old[int(ix)])
    additions = budget - deletes
    attempts = 0
    while additions and attempts < max(200, 40 * budget):
        attempts += 1
        a, b = rng.integers(0, n, size=2).tolist()
        if a == b:
            continue
        pair = (min(a, b), max(a, b))
        if pair not in pairs:
            pairs.add(pair)
            additions -= 1
    if additions:
        raise ValueError('Cannot complete edge additions (graph may be dense).')
    return torch.tensor(sorted(pairs), device=edges.device, dtype=torch.long).T.contiguous()


def aligned_mask_drift(edges_a, scores_a, edges_b, scores_b):
    """Differentiable MSE on shared undirected edges; does not penalize absent edges."""
    a = {tuple(p): i for i, p in enumerate(edges_a.T.detach().cpu().tolist())}
    b = {tuple(p): i for i, p in enumerate(edges_b.T.detach().cpu().tolist())}
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return scores_a.sum() * 0
    ia = torch.tensor([a[k] for k in shared], device=scores_a.device)
    ib = torch.tensor([b[k] for k in shared], device=scores_b.device)
    return (scores_a[ia] - scores_b[ib]).square().mean()


def evaluate_drift(model, x, edges, y, sensitive, test_ids, budget, steps, seed):
    """Sequential random edits. Reports final and worst per-step changes.

    This is a cumulative stress test, not an optimized adversarial attack.
    """
    model.eval()
    with torch.no_grad():
        base_logits, base_mask = model(x, edges)
        base_pred = base_logits.argmax(-1)[test_ids]
        base_prob = base_logits.softmax(-1)[test_ids]
        original = {tuple(p): float(v) for p, v in zip(edges.T.cpu().tolist(), base_mask.cpu().tolist())}
        current = edges
        changes, kls, mask_changes, f1s = [], [], [], []
        for t in range(steps):
            current = edit_edges(current, len(x), budget, seed * 1000 + t)
            logits, mask = model(x, current)
            pred = logits.argmax(-1)[test_ids]
            changes.append(float((pred != base_pred).float().mean()))
            kls.append(float(F.kl_div(logits[test_ids].log_softmax(-1),base_prob,reduction='batchmean')))
            now = {tuple(p): float(v) for p,v in zip(current.T.cpu().tolist(),mask.cpu().tolist())}
            common = original.keys() & now.keys()
            mask_changes.append(float(np.mean([abs(original[k]-now[k]) for k in common])) if common else float('nan'))
            f1s.append(float(f1_score(y[test_ids].cpu().numpy(),pred.cpu().numpy(),zero_division=0)))
    return {'drift_pred_change_final':changes[-1], 'drift_pred_change_max':max(changes),
            'drift_kl_final':kls[-1], 'drift_mask_l1_final':mask_changes[-1],
            'drift_test_f1_final':f1s[-1]}
