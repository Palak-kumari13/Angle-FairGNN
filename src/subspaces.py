"""Rank-aware principal angles for row-sample matrices in one common space."""
import numpy as np


def basis(samples, variance=0.95, max_rank=12, tol=1e-9):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('samples must be a finite 2D array')
    if not x.size or np.linalg.norm(x) < tol:
        return np.zeros((x.shape[1], 0)), np.array([])
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(singular > max(tol, singular[0] * 1e-6)))
    if rank == 0:
        return np.zeros((x.shape[1], 0)), singular
    fractions = np.cumsum(singular[:rank] ** 2) / np.sum(singular[:rank] ** 2)
    r = min(rank, max_rank, int(np.searchsorted(fractions, variance) + 1))
    return vt[:r].T.copy(), singular


def angles(a, b):
    if a.shape[0] != b.shape[0]:
        raise ValueError('bases must share the same feature space')
    if a.shape[1] == 0 or b.shape[1] == 0:
        return np.array([]), float('nan')
    cs = np.linalg.svd(a.T @ b, compute_uv=False)
    cs = np.clip(cs, 0., 1.)
    return np.degrees(np.arccos(cs)), float(np.sum(cs ** 2) / min(a.shape[1], b.shape[1]))


def selective_basis(a, t, tol=1e-5):
    if a.shape[1] == 0:
        return a
    residual = a - t @ (t.T @ a)
    u, s, _ = np.linalg.svd(residual, full_matrices=False)
    r = int(np.sum(s > max(tol, s[0] * 1e-4))) if len(s) else 0
    return u[:, :r].copy()


def whitened_angles(sensitive, task, ridge=0.1, variance=.95, max_rank=12):
    """Sample-covariance metric, restricted to the joint drift span.

    Reports a sensitivity analysis, not a canonical or causal metric.
    """
    joined = np.vstack([sensitive, task]).astype(np.float64)
    if not joined.size or not np.linalg.norm(joined):
        return np.array([]), float('nan')
    _, _, vt = np.linalg.svd(joined, full_matrices=False)
    rank = min(joined.shape[0], int(np.linalg.matrix_rank(joined)))
    if rank == 0:
        return np.array([]), float('nan')
    coords = joined @ vt[:rank].T
    covariance = (coords.T @ coords) / len(joined)
    ev, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(ev)), 1e-12)
    whitening = (vectors * (ev + ridge * scale) ** -.5) @ vectors.T
    n = len(sensitive)
    ua, _ = basis(coords[:n] @ whitening, variance, max_rank)
    ut, _ = basis(coords[n:] @ whitening, variance, max_rank)
    return angles(ua, ut)


def null_overlaps(a, b, repetitions=100, seed=0):
    """Same-rank Haar random subspaces in the original ambient space."""
    rng = np.random.default_rng(seed)
    d, rank_b = b.shape
    output = []
    for _ in range(repetitions):
        q, _ = np.linalg.qr(rng.normal(size=(d, rank_b)), mode='reduced')
        output.append(angles(a, q)[1])
    return np.array(output)
