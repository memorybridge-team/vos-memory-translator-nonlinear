"""FP64 centered sufficient statistics; row-vector affine convention."""
from dataclasses import dataclass
import math

import torch


class Moments:
    def __init__(self, dx, dy):
        self.n = 0
        self.mx = torch.zeros(dx, dtype=torch.float64)
        self.my = torch.zeros(dy, dtype=torch.float64)
        self.xx = torch.zeros(dx, dx, dtype=torch.float64)
        self.xy = torch.zeros(dx, dy, dtype=torch.float64)
        self.yy = torch.zeros(dy, dtype=torch.float64)

    def update(self, x, y):
        x, y = x.to(device="cpu", dtype=torch.float64), y.to(device="cpu", dtype=torch.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("expected corresponding [N,D] matrices")
        if x.shape[1] != self.mx.numel() or y.shape[1] != self.my.numel():
            raise ValueError("feature dimension mismatch")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("nonfinite observations")
        if not len(x):
            return
        other = Moments(x.shape[1], y.shape[1])
        other.n = len(x)
        other.mx, other.my = x.mean(0), y.mean(0)
        xc, yc = x - other.mx, y - other.my
        other.xx, other.xy = xc.T @ xc, xc.T @ yc
        other.yy = yc.square().sum(0)
        self.merge(other)

    def merge(self, other):
        if not other.n:
            return
        if self.xx.shape != other.xx.shape or self.xy.shape != other.xy.shape:
            raise ValueError("moment dimension mismatch")
        n = self.n + other.n
        dx, dy = other.mx - self.mx, other.my - self.my
        correction = self.n * other.n / n
        self.xx += other.xx + torch.outer(dx, dx) * correction
        self.xy += other.xy + torch.outer(dx, dy) * correction
        self.yy += other.yy + dy.square() * correction
        self.mx += dx * (other.n / n)
        self.my += dy * (other.n / n)
        self.n = n

    def payload(self):
        return dict(n=self.n, mx=self.mx, my=self.my, xx=self.xx, xy=self.xy, yy=self.yy)

    @classmethod
    def from_payload(cls, payload):
        obj = cls(len(payload["mx"]), len(payload["my"]))
        obj.n = int(payload["n"])
        if obj.n < 1:
            raise ValueError("empty statistics")
        for name in ("mx", "my", "xx", "xy", "yy"):
            value = payload[name]
            if value.shape != getattr(obj, name).shape or value.dtype != torch.float64:
                raise ValueError("invalid FP64 statistics")
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite statistics")
            setattr(obj, name, value)
        return obj

    def rms2(self):
        return float((self.yy / self.n + self.my.square()).mean())


@dataclass
class Affine:
    weight: torch.Tensor
    bias: torch.Tensor
    diagnostics: dict

    def __call__(self, x):
        return x @ self.weight.to(x) + self.bias.to(x)


def solve(stats, alpha, rank_rtol=1e-12):
    if stats.n < 1 or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("nonempty observations and finite nonnegative alpha required")
    if not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must be in (0,1)")
    gram = (stats.xx + stats.xx.T) / (2 * stats.n)
    cross = stats.xy / stats.n
    values, vectors = torch.linalg.eigh(gram)
    largest = float(values[-1])
    scale = float(gram.trace() / gram.shape[0])
    if largest <= 0 or scale <= 0:
        raise ValueError("degenerate input: zero variance")
    # Cutoff is relative to Gram eigenvalues, not singular values of X.
    cutoff = largest * rank_rtol
    if float(values[0]) < -largest * 1e-10:
        raise ValueError("Gram matrix is not positive semidefinite")
    keep = values > cutoff
    penalty = float(alpha * scale)
    if alpha == 0:
        basis = vectors[:, keep]
        weight = basis @ ((basis.T @ cross) / values[keep, None])
        solver = "eigh_minimum_norm"
    else:
        chol, info = torch.linalg.cholesky_ex(gram + penalty * torch.eye(len(gram), dtype=gram.dtype))
        if int(info) != 0:
            raise ValueError("Cholesky failed; alpha/jitter were not changed")
        weight = torch.cholesky_solve(cross, chol)
        solver = "cholesky"
    bias = stats.my - stats.mx @ weight
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("nonfinite affine solution")
    diagnostic = dict(alpha=float(alpha), lambda_value=penalty, variance_scale=scale,
        observations=stats.n, solver=solver, rank=int(keep.sum()), rank_rtol=rank_rtol,
        cutoff_domain="normalized_gram_eigenvalues", cutoff=cutoff,
        gram_eigenvalues=values.tolist(), retained_condition=largest / float(values[keep][0]),
        full_condition=largest / float(values[0]) if float(values[0]) > cutoff else None,
        weight_singular_values=torch.linalg.svdvals(weight).tolist())
    return Affine(weight, bias, diagnostic)


def per_feature_sse(stats, affine):
    w, b = affine.weight.double(), affine.bias.double()
    residual_mean = stats.mx @ w + b - stats.my
    sse = (w * (stats.xx @ w)).sum(0) - 2 * (w * stats.xy).sum(0) + stats.yy
    sse += stats.n * residual_mean.square()
    tolerance = 1e-8 * max(float(stats.yy.abs().max()), 1.)
    if float(sse.min()) < -tolerance:
        raise ValueError("negative SSE beyond numerical tolerance")
    return sse.clamp_min(0)


def metrics(stats, affine, fit_rms2, r2_variance_floor=1e-12):
    sse = per_feature_sse(stats, affine)
    keep = stats.yy / stats.n > r2_variance_floor
    mse = float(sse.sum() / (stats.n * len(stats.my)))
    return dict(mse=mse, normalized_mse=mse / max(fit_rms2, 1e-12),
        r2_variance_weighted=1 - float(sse[keep].sum() / stats.yy[keep].sum()) if keep.any() else None,
        r2_excluded_coordinates=int((~keep).sum()), r2_variance_floor=r2_variance_floor,
        observations=stats.n, target_rms2=stats.rms2())


def select_alpha(candidates, relative_tie=1e-8):
    best = min(row["development_mse"] for row in candidates)
    tolerance = relative_tie * max(abs(best), torch.finfo(torch.float64).tiny)
    return min((r for r in candidates if r["development_mse"] <= best + tolerance), key=lambda r: r["alpha"])
