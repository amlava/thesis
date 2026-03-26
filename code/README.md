## SIPM

Minimal PyTorch implementation of the Stochastic Inexact Penalty Method (SIPM).

### Install

```bash
uv sync
```

### Dev install

```bash
uv sync --extra dev
```

### Pre-commit

```bash
uv run pre-commit install
```

### Run tests

```bash
uv run pytest -q
```

### Quick start

```python
import torch
from sipm import SIPMConfig, StochasticInexactPenaltyMethod, constant_schedule


def sampler(batch_size: int):
    zeta = torch.zeros(batch_size, 1)
    xi = torch.zeros(batch_size, 1)
    return zeta, xi


def constraint_map(xi_batch):
    batch = xi_batch.shape[0]
    n = 4
    a = torch.zeros(batch, 1, n)
    a[:, 0, 0] = 1.0
    b = torch.zeros(batch, 1)
    return a, b


def objective(x, zeta_batch):
    batch = zeta_batch.shape[0]
    loss = 0.5 * (x[1:] ** 2).sum()
    return loss.repeat(batch)


cfg = SIPMConfig(
    num_iters=500,
    step_size=constant_schedule(0.1),
    penalty_weight=constant_schedule(1.0),
    delta=constant_schedule(1.0),
    iterate_weight=constant_schedule(1.0),
    batch_size=1,
)

solver = StochasticInexactPenaltyMethod(
    objective=objective,
    constraint_map=constraint_map,
    sampler=sampler,
    config=cfg,
)

x0 = torch.zeros(4)
x_bar, history = solver.run(x0)
print(x_bar, history["violation"][-1])
```
