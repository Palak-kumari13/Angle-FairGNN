"""Global, validation-only fallback controller for standalone AngleFair-GNN.

This decides which *already trained* checkpoint to deploy. It never infers
success from test metrics, never relabels heuristic masks as learned explanations,
and never asserts fairness/stability without measurements.
"""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class GateConfig:
    max_dp: float = 0.15
    max_eo: float = 0.15
    max_auc_drop: float = 0.03
    max_prediction_drift: float = 0.08
    min_mask_std: float = 1e-4
    min_auc: float = 0.5


def finite(v):
    try:
        return math.isfinite(float(v))
    except (ValueError, TypeError):
        return False


def choose_checkpoint(rows, config=GateConfig()):
    """Select valid checkpoint and independently decide whether masks may be reported.

    Each row must contain val_auc, val_parity, val_equality, val_prediction_drift.
    Mask validation status is reported as *not certified* until a separate
    matched-random and fidelity evaluation is available.
    """
    by_name={r['model']:r for r in rows}
    if 'A0' not in by_name:
        return {'selected_model': None, 'decision':'no_reference', 'explanation':'abstain',
                'reason':['Missing A0 reference checkpoint'], 'reference_passed':False}
    reference=by_name['A0']
    reasons={}
    ref_auc=reference.get('val_auc')
    eligible=[]
    for name, row in by_name.items():
        failures=[]
        auc=row.get('val_auc')
        if not finite(auc) or float(auc)<config.min_auc:failures.append('invalid_or_low_auc')
        if finite(ref_auc) and finite(auc) and float(auc)<float(ref_auc)-config.max_auc_drop:
            failures.append('utility_regression')
        for key, limit in [('val_parity',config.max_dp),('val_equality',config.max_eo),
                           ('val_prediction_drift',config.max_prediction_drift)]:
            value=row.get(key)
            if not finite(value) or float(value)>limit:failures.append(key+'_failed_or_missing')
        reasons[name]=failures
        if not failures:eligible.append(row)
    if eligible:
        # Select highest validation AUC among candidates satisfying all constraints.
        # Prefer unmasked reference if differences are numerically indistinguishable.
        best=max(eligible,key=lambda r:(float(r['val_auc']),r['model']=='A0'))
        name=best['model']; decision='accepted' if name!='A0' else 'reference_selected'
    else:
        name='A0';decision='no_certified_candidate'
    explanation='abstain'
    # No explanation is certified by a mere score-variance test.
    reasons_for_mask=[]
    if name!='A0':
        row=by_name[name]
        if not finite(row.get('val_mask_std')) or float(row['val_mask_std'])<config.min_mask_std:
            reasons_for_mask.append('collapsed_mask')
        reasons_for_mask.append('matched_random_fidelity_not_certified')
    return {'selected_model':name,'decision':decision,'explanation':explanation,
            'reason':reasons.get(name,[]),'candidate_failures':reasons,
            'mask_reasons':reasons_for_mask, 'reference_passed':not bool(reasons['A0']),
            'thresholds':asdict(config)}
