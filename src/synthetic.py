"""Controlled synthetic graph with planted task and sensitive feature directions."""
import numpy as np
from sklearn.model_selection import train_test_split


def make_graph(n=240, degree=8, angle=45, seed=0, correlation=0.0, signal=1.4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n, dtype=np.int64)
    a = (rng.random(n) < np.where(y == 1, .5 + correlation / 2, .5 - correlation / 2)).astype(np.int64)
    theta = np.radians(angle)
    task = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
    sensitive = np.array([np.cos(theta), np.sin(theta), 0., 0., 0., 0., 0., 0.])
    x = rng.normal(0, 1, (n, 8)) + signal * (2 * y[:, None] - 1) * task + signal * (2 * a[:, None] - 1) * sensitive
    # Labels and sensitive-attribute homophily fixed across angle settings.
    pairs = []
    for i in range(n):
        candidates = np.arange(i + 1, n)
        if not len(candidates):
            continue
        weights = np.exp(.75 * (y[candidates] == y[i]) + .35 * (a[candidates] == a[i]))
        probs = np.minimum((degree / (n-1)) * weights / 1.65, 1)
        added = candidates[rng.random(len(candidates)) < probs]
        pairs.extend((i, int(j)) for j in added)
    edges = np.asarray(pairs, dtype=np.int64).T
    idx = np.arange(n)
    tr, other = train_test_split(idx, train_size=.6, random_state=42, stratify=y * 2 + a)
    va, te = train_test_split(other, test_size=.5, random_state=42, stratify=y[other] * 2 + a[other])
    return dict(x=x.astype('float32'), edge_index=edges, y=y, sensitive=a,
                train_idx=np.sort(tr), val_idx=np.sort(va), test_idx=np.sort(te),
                true_angle=np.array([angle], dtype=float))
