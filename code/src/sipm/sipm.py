from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol, Tuple, Union

import torch

Tensor = torch.Tensor
Sample = Union[Tensor, Tuple["Sample", ...], list, dict, None]


class ProxOperator(Protocol):
    def prox(self, x: Tensor, step: float) -> Tensor:
        ...


class ZeroProx:
    def prox(self, x: Tensor, step: float) -> Tensor:
        return x


@dataclass(frozen=True)
class L1Prox:
    lam: float

    def prox(self, x: Tensor, step: float) -> Tensor:
        thresh = self.lam * step
        return torch.sign(x) * torch.clamp(x.abs() - thresh, min=0.0)


@dataclass(frozen=True)
class L2SquaredProx:
    lam: float

    def prox(self, x: Tensor, step: float) -> Tensor:
        return x / (1.0 + 2.0 * self.lam * step)


@dataclass(frozen=True)
class ElasticNetProx:
    l1: float
    l2: float

    def prox(self, x: Tensor, step: float) -> Tensor:
        x = torch.sign(x) * torch.clamp(x.abs() - self.l1 * step, min=0.0)
        return x / (1.0 + 2.0 * self.l2 * step)


class NonNegativityProx:
    def prox(self, x: Tensor, step: float) -> Tensor:
        return torch.clamp(x, min=0.0)


@dataclass(frozen=True)
class BoxProx:
    low: float
    high: float

    def prox(self, x: Tensor, step: float) -> Tensor:
        return torch.clamp(x, min=self.low, max=self.high)


@dataclass(frozen=True)
class SimplexProx:
    radius: float = 1.0

    def prox(self, x: Tensor, step: float) -> Tensor:
        # Euclidean projection onto the probability simplex {x >= 0, sum x = radius}
        if x.numel() == 0:
            return x
        v = x.view(-1)
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0) - self.radius
        rho = torch.nonzero(v_sorted - cssv / (torch.arange(1, v.numel() + 1, device=v.device)) > 0, as_tuple=False)
        if rho.numel() == 0:
            theta = 0.0
        else:
            rho_idx = int(rho[-1].item()) + 1
            theta = cssv[rho_idx - 1] / rho_idx
        w = torch.clamp(v - theta, min=0.0)
        return w.view_as(x)


@dataclass(frozen=True)
class SumToOneProx:
    target: float = 1.0

    def prox(self, x: Tensor, step: float) -> Tensor:
        # Euclidean projection onto {x : sum x = target}
        if x.numel() == 0:
            return x
        offset = (x.sum() - self.target) / x.numel()
        return x - offset


Schedule = Callable[[int], float]
Sampler = Callable[[int], Tuple[object, object]]


class BatchSampler(Protocol):
    def __call__(self, batch_size: int) -> Tuple[object, object]:
        ...


class DistributionSampler:
    def __init__(self, sample_fn: Callable[[int], Tuple[object, object]]) -> None:
        self._sample_fn = sample_fn

    def __call__(self, batch_size: int) -> Tuple[object, object]:
        return self._sample_fn(batch_size)


class DatasetSampler:
    def __init__(
        self,
        dataset: Iterable,
        *,
        drop_last: bool = False,
        cycle: bool = True,
    ) -> None:
        self._dataset = dataset
        self._drop_last = drop_last
        self._cycle = cycle
        self._iterator = iter(dataset)

    def __call__(self, batch_size: int) -> Tuple[object, object]:
        batch = []
        while len(batch) < batch_size:
            try:
                item = next(self._iterator)
            except StopIteration:
                if not self._cycle:
                    break
                self._iterator = iter(self._dataset)
                item = next(self._iterator)
            batch.append(item)

        if len(batch) < batch_size and self._drop_last:
            raise StopIteration("Dataset exhausted before full batch.")

        zeta_list, xi_list = zip(*batch)
        zeta = torch.utils.data.default_collate(list(zeta_list))
        xi = torch.utils.data.default_collate(list(xi_list))
        return zeta, xi


def make_sampler(
    source: Union[Callable[[int], Tuple[object, object]], Iterable],
) -> BatchSampler:
    if callable(source):
        return DistributionSampler(source)
    return DatasetSampler(source)
Objective = Callable[[Tensor, object], Tensor]
ConstraintMap = Callable[[object], Tuple[Tensor, Tensor]]


def constant_schedule(value: float) -> Schedule:
    def _fn(_: int) -> float:
        return float(value)

    return _fn


def polynomial_schedule(base: float, power: float) -> Schedule:
    def _fn(k: int) -> float:
        return float(base * (k + 1) ** (-power))

    return _fn


def _ensure_vector_loss(loss: Tensor) -> Tensor:
    if loss.ndim == 0:
        return loss.unsqueeze(0)
    return loss


def _move_sample_to_device(sample: Sample, device: torch.device, dtype: torch.dtype) -> Sample:
    if isinstance(sample, Tensor):
        if sample.is_floating_point():
            return sample.to(device=device, dtype=dtype)
        return sample.to(device=device)
    if isinstance(sample, tuple):
        return tuple(_move_sample_to_device(s, device, dtype) for s in sample)
    if isinstance(sample, list):
        return [_move_sample_to_device(s, device, dtype) for s in sample]
    if isinstance(sample, dict):
        return {k: _move_sample_to_device(v, device, dtype) for k, v in sample.items()}
    return sample


def huber_penalty_and_grad(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    delta: float,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """
    Vectorized Huber-like penalty and gradient.

    Args:
        x: shape (n,)
        a: shape (batch, m, n)
        b: shape (batch, m)
    Returns:
        penalty: shape (batch,)
        grad: shape (n,)
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor (n,)")
    if a.ndim != 3 or b.ndim != 2:
        raise ValueError("a must be (batch, m, n) and b must be (batch, m)")

    batch, m, n = a.shape
    if b.shape != (batch, m):
        raise ValueError("b must have shape (batch, m)")

    s = torch.einsum("bmn,n->bm", a, x) + b
    row_norm = torch.linalg.norm(a, dim=2).clamp_min(eps)
    s_over_norm = s / row_norm

    delta_t = torch.as_tensor(delta, dtype=x.dtype, device=x.device)

    above = s > delta_t
    mid = (s >= -delta_t) & (s <= delta_t)

    penalty = torch.zeros_like(s_over_norm)
    penalty = torch.where(above, s_over_norm, penalty)
    mid_term = (s + delta_t) ** 2 / (4.0 * delta_t * row_norm)
    penalty = torch.where(mid, mid_term, penalty)

    grad_coeff = torch.zeros_like(s)
    grad_coeff = torch.where(above, 1.0 / row_norm, grad_coeff)
    mid_coeff = (s + delta_t) / (2.0 * delta_t * row_norm)
    grad_coeff = torch.where(mid, mid_coeff, grad_coeff)

    grad = torch.einsum("bmn,bm->n", a, grad_coeff) / float(m * batch)
    return penalty.mean(dim=1), grad


def constraint_violation(
    x: Tensor, a: Tensor, b: Tensor, eps: float = 1e-12
) -> Tensor:
    """
    Returns per-sample average violation: mean_i max(0, (a_i x - b_i) / ||a_i||).
    Shape: (batch,)
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor (n,)")
    if a.ndim != 3 or b.ndim != 2:
        raise ValueError("a must be (batch, m, n) and b must be (batch, m)")
    batch, m, _ = a.shape
    if b.shape != (batch, m):
        raise ValueError("b must have shape (batch, m)")

    s = torch.einsum("bmn,n->bm", a, x) + b
    row_norm = torch.linalg.norm(a, dim=2).clamp_min(eps)
    violation = torch.relu(s) / row_norm
    return violation.mean(dim=1)


@dataclass
class SIPMConfig:
    num_iters: int
    step_size: Schedule
    penalty_weight: Schedule
    delta: Schedule
    iterate_weight: Schedule
    batch_size: int = 1
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    move_samples_to_device: bool = True


class StochasticInexactPenaltyMethod:
    def __init__(
        self,
        objective: Objective,
        constraint_map: ConstraintMap,
        sampler: Sampler,
        prox: Optional[ProxOperator] = None,
        config: Optional[SIPMConfig] = None,
    ) -> None:
        self.objective = objective
        self.constraint_map = constraint_map
        self.sampler = sampler
        self.prox = prox or ZeroProx()
        if config is None:
            raise ValueError("config is required")
        self.config = config

    def run(self, x0: Tensor) -> Tuple[Tensor, dict]:
        cfg = self.config
        device = torch.device(cfg.device)
        x = x0.detach().to(device=device, dtype=cfg.dtype).clone()

        weight_sum = 0.0
        x_bar = torch.zeros_like(x)

        history = {
            "x": [],
            "obj": [],
            "penalty": [],
            "violation": [],
        }

        for k in range(cfg.num_iters):
            step = cfg.step_size(k)
            gamma = cfg.penalty_weight(k)
            delta = cfg.delta(k)
            weight = cfg.iterate_weight(k + 1)

            zeta_batch, xi_batch = self.sampler(cfg.batch_size)
            if cfg.move_samples_to_device:
                zeta_batch = _move_sample_to_device(zeta_batch, device, cfg.dtype)
                xi_batch = _move_sample_to_device(xi_batch, device, cfg.dtype)
            a, b = self.constraint_map(xi_batch)

            a = a.to(device=device, dtype=cfg.dtype)
            b = b.to(device=device, dtype=cfg.dtype)

            x_req = x.detach().clone().requires_grad_(True)
            loss = _ensure_vector_loss(self.objective(x_req, zeta_batch))
            loss = loss.mean()
            if loss.requires_grad:
                loss.backward()
                grad_f = x_req.grad.detach()
            else:
                grad_f = torch.zeros_like(x)

            penalty, grad_h = huber_penalty_and_grad(x, a, b, delta)
            violation = constraint_violation(x, a, b)
            grad = grad_f + gamma * grad_h

            with torch.no_grad():
                x = self.prox.prox(x - step * grad, step)

            weight_sum += float(weight)
            x_bar = x_bar + float(weight) * x

            history["x"].append(x.detach().clone())
            history["obj"].append(float(loss.detach()))
            history["penalty"].append(float(penalty.mean().detach()))
            history["violation"].append(float(violation.mean().detach()))

        x_bar = x_bar / weight_sum
        return x_bar, history


__all__ = [
    "SIPMConfig",
    "StochasticInexactPenaltyMethod",
    "constant_schedule",
    "polynomial_schedule",
    "BatchSampler",
    "DistributionSampler",
    "DatasetSampler",
    "make_sampler",
    "ZeroProx",
    "L1Prox",
    "L2SquaredProx",
    "ElasticNetProx",
    "NonNegativityProx",
    "BoxProx",
]
