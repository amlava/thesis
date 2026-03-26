import math

import torch

from sipm import (
    SIPMConfig,
    StochasticInexactPenaltyMethod,
    constant_schedule,
    DatasetSampler,
    DistributionSampler,
)


def _make_constraint_map(n: int):
    def constraint_map(xi_batch):
        batch = xi_batch.shape[0]
        a = torch.zeros(batch, 1, n, dtype=xi_batch.dtype, device=xi_batch.device)
        a[:, 0, 0] = 1.0
        b = torch.zeros(batch, 1, dtype=xi_batch.dtype, device=xi_batch.device)
        return a, b

    return constraint_map


def _sampler(batch_size: int):
    zeta = torch.zeros(batch_size, 1)
    xi = torch.zeros(batch_size, 1)
    return zeta, xi


def _convex_step_size(k: int) -> float:
    if k == 0:
        return 1.0 / (math.log(2.0) ** 2)
    return 1.0 / ((math.log(k + 2.0) ** 2) * math.sqrt(k))


def _convex_delta(k: int) -> float:
    if k == 0:
        return 1.0
    return 1.0 / k


def _convex_penalty(k: int) -> float:
    return math.log(k + 1.0)


def _strong_step_size(mu: float, k: int) -> float:
    if k == 0:
        return 2.0 / mu
    return 2.0 / (mu * k)


def _strong_delta(k: int) -> float:
    if k == 0:
        return 1.0
    return 1.0 / k


def _strong_penalty(k: int) -> float:
    return math.log(k + 1.0)


def test_convex_feasibility_decreases():
    torch.manual_seed(0)
    n = 4
    c = torch.tensor([2.0, -1.0, 0.5, 1.5])

    def objective(x, zeta_batch):
        batch = zeta_batch.shape[0]
        loss = 0.5 * (x[1:] - c[1:]).pow(2).sum()
        return loss.repeat(batch)

    cfg = SIPMConfig(
        num_iters=1000,
        step_size=_convex_step_size,
        penalty_weight=_convex_penalty,
        delta=_convex_delta,
        iterate_weight=_convex_step_size,
        batch_size=1,
        device="cpu",
    )

    solver = StochasticInexactPenaltyMethod(
        objective=objective,
        constraint_map=_make_constraint_map(n),
        sampler=_sampler,
        config=cfg,
    )

    x0 = torch.zeros(n)
    _, history = solver.run(x0)
    final_penalty = history["penalty"][-1]

    assert final_penalty < 1e-1


def test_strongly_convex_distance_decreases():
    torch.manual_seed(0)
    n = 3
    mu = 2.0
    c = torch.tensor([1.5, -2.0, 0.5])

    def objective(x, zeta_batch):
        batch = zeta_batch.shape[0]
        loss = 0.5 * mu * (x - c).pow(2).sum()
        return loss.repeat(batch)

    cfg = SIPMConfig(
        num_iters=1000,
        step_size=lambda k: _strong_step_size(mu, k),
        penalty_weight=_strong_penalty,
        delta=_strong_delta,
        iterate_weight=lambda k: 1.0 / _strong_step_size(mu, k),
        batch_size=1,
        device="cpu",
    )

    solver = StochasticInexactPenaltyMethod(
        objective=objective,
        constraint_map=_make_constraint_map(n),
        sampler=_sampler,
        config=cfg,
    )

    x0 = torch.zeros(n)
    x_bar, _ = solver.run(x0)

    x_star = c.clone()
    x_star[0] = min(c[0].item(), 0.0)
    dist = torch.linalg.norm(x_bar - x_star).item()

    assert dist < 0.1


def test_device_handling_gpu_if_available():
    if not torch.cuda.is_available():
        return

    n = 2
    c = torch.tensor([1.0, -1.0], device="cuda")

    def objective(x, zeta_batch):
        batch = zeta_batch.shape[0]
        loss = 0.5 * (x - c).pow(2).sum()
        return loss.repeat(batch)

    cfg = SIPMConfig(
        num_iters=10,
        step_size=constant_schedule(0.1),
        penalty_weight=constant_schedule(1.0),
        delta=constant_schedule(1.0),
        iterate_weight=constant_schedule(1.0),
        batch_size=1,
        device="cuda",
    )

    solver = StochasticInexactPenaltyMethod(
        objective=objective,
        constraint_map=_make_constraint_map(n),
        sampler=_sampler,
        config=cfg,
    )

    x0 = torch.zeros(n)
    x_bar, _ = solver.run(x0)
    assert x_bar.is_cuda


def test_distribution_sampler_passthrough():
    def sample_fn(batch_size: int):
        zeta = torch.arange(batch_size).float().unsqueeze(1)
        xi = torch.arange(batch_size).float().unsqueeze(1) + 10.0
        return zeta, xi

    sampler = DistributionSampler(sample_fn)
    zeta, xi = sampler(3)

    assert zeta.shape == (3, 1)
    assert xi.shape == (3, 1)
    assert torch.allclose(zeta.squeeze(), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(xi.squeeze(), torch.tensor([10.0, 11.0, 12.0]))


def test_dataset_sampler_batches_and_cycles():
    zeta = torch.tensor([[1.0], [2.0]])
    xi = torch.tensor([[3.0], [4.0]])
    dataset = list(zip(zeta, xi))

    sampler = DatasetSampler(dataset, cycle=True)
    z_batch, x_batch = sampler(3)

    assert z_batch.shape == (3, 1)
    assert x_batch.shape == (3, 1)
    assert torch.allclose(z_batch.squeeze(), torch.tensor([1.0, 2.0, 1.0]))
    assert torch.allclose(x_batch.squeeze(), torch.tensor([3.0, 4.0, 3.0]))
