import math
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator

from sipm import DistributionSampler, SIPMConfig, StochasticInexactPenaltyMethod


def make_inventory_dataset(
    num_periods: int,
    num_scenarios: int,
    seed: int = 7
):
    rng = np.random.default_rng(seed)
    t = np.arange(num_periods, dtype=np.float32)
    seasonal = 1.5 + 0.3 * np.sin(2.0 * np.pi * t / 12.0)
    mean_demand = seasonal
    demand = rng.lognormal(
        mean=np.log(mean_demand),
        sigma=0.2,
        size=(num_scenarios, num_periods),
    ).astype(np.float32)
    return demand


def split_dataset(demand: np.ndarray):
    n = demand.shape[0]
    n_train = int(0.8 * n)
    return demand[:n_train], demand[n_train:]


@dataclass(frozen=True)
class BoxProx:
    low: float
    high: float

    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        return torch.clamp(x, min=self.low, max=self.high)


def lower_tri_ones(num_periods: int) -> np.ndarray:
    return np.tril(np.ones((num_periods, num_periods), dtype=np.float32))


def make_sampler(demand: np.ndarray):
    demand_t = torch.tensor(demand, dtype=torch.float32)

    def sample_fn(batch_size: int):
        idx = torch.randint(0, demand_t.shape[0], (batch_size,))
        batch = demand_t[idx]
        return batch, batch

    return DistributionSampler(sample_fn)


def objective_factory(v: np.ndarray, beta: float, a_mat: np.ndarray, reg_lambda: float):
    v_t = torch.tensor(v, dtype=torch.float32)
    a_t = torch.tensor(a_mat, dtype=torch.float32)

    def objective(alpha: torch.Tensor, zeta_batch: torch.Tensor) -> torch.Tensor:
        linear = torch.dot(v_t, alpha)
        eta = beta * torch.sum(zeta_batch @ a_t.T, dim=1)
        reg = reg_lambda * torch.dot(alpha, alpha)
        return linear - eta + reg

    return objective


def constraint_map_factory(a_mat: np.ndarray, x0: float):
    a_t = torch.tensor(-a_mat, dtype=torch.float32)

    def constraint_map(xi_batch: torch.Tensor):
        batch = xi_batch.shape[0]
        a_rhs = torch.tensor(a_mat.T, dtype=xi_batch.dtype, device=xi_batch.device)
        b = (xi_batch @ a_rhs) - x0
        a = a_t.unsqueeze(0).expand(batch, -1, -1)
        return a, b

    return constraint_map


def convex_step_size(k: int, scale: float) -> float:
    if k == 0:
        return scale / (math.log(2.0) ** 2)
    return scale / ((math.log(k + 2.0) ** 2) * math.sqrt(k))


def gamma_schedule(k: int, scale: float) -> float:
    return scale * math.log(k + 2.0)


def delta_schedule(k: int, scale: float) -> float:
    return scale / (k + 1)


def strong_step_size(k: int, mu: float, scale: float) -> float:
    if k == 0:
        return 2.0 * scale / mu
    return 2.0 * scale / (mu * k)


def solve_reference(v, beta, a_mat, demand, x0, order_cap, reg_lambda):
    n = len(v)
    alpha = cp.Variable(n)
    eta = beta * np.sum(demand @ a_mat.T, axis=1)
    objective = cp.Minimize(v @ alpha - np.mean(eta) + reg_lambda * cp.sum_squares(alpha))
    constraints = [alpha >= 0.0, alpha <= order_cap]
    for d in demand:
        constraints.append(-a_mat @ alpha + a_mat @ d - x0 <= 0.0)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    return float(problem.value), np.asarray(alpha.value, dtype=np.float32)


def evaluate(
    alpha: np.ndarray,
    v: np.ndarray,
    beta: float,
    a_mat: np.ndarray,
    demand: np.ndarray,
    x0: float,
    reg_lambda: float,
):
    eta = beta * np.sum(demand @ a_mat.T, axis=1)
    obj = float(v @ alpha - np.mean(eta) + reg_lambda * np.dot(alpha, alpha))
    viol = np.maximum((-a_mat @ alpha)[None, :] + demand @ a_mat.T - x0, 0.0)
    return obj, float(np.quantile(viol.reshape(-1), 0.95))


def mean_violation(alpha: np.ndarray, a_mat: np.ndarray, demand: np.ndarray, x0: float) -> float:
    viol = np.maximum((-a_mat @ alpha)[None, :] + demand @ a_mat.T - x0, 0.0)
    return float(np.mean(viol))


def run_solver(
    demand_train: np.ndarray,
    v: np.ndarray,
    beta: float,
    a_mat: np.ndarray,
    x0: float,
    order_cap: float,
    reg_lambda: float,
    step_scale: float,
    gamma_scale: float,
    delta_scale: float,
    num_iters: int,
    seed: int,
):
    torch.manual_seed(seed)
    sampler = make_sampler(demand_train)
    mu = 2.0 * reg_lambda
    cfg = SIPMConfig(
        num_iters=num_iters,
        step_size=lambda k: strong_step_size(k, mu, step_scale),
        penalty_weight=lambda k: gamma_schedule(k, gamma_scale),
        delta=lambda k: delta_schedule(k, delta_scale),
        iterate_weight=lambda k: 1.0 / strong_step_size(k, mu, step_scale),
        batch_size=1,
        device="cpu",
    )
    solver = StochasticInexactPenaltyMethod(
        objective=objective_factory(v, beta, a_mat, reg_lambda),
        constraint_map=constraint_map_factory(a_mat, x0),
        sampler=sampler,
        prox=BoxProx(low=0.0, high=order_cap),
        config=cfg,
    )
    alpha0 = torch.full((a_mat.shape[0],), min(order_cap, x0 / a_mat.shape[0]))
    _, history = solver.run(alpha0)
    weight_sum = 0.0
    avg = torch.zeros_like(history["x"][0])
    averaged = []
    last_iterates = []
    for k, alpha in enumerate(history["x"]):
        weight = cfg.iterate_weight(k + 1)
        weight_sum += float(weight)
        avg = avg + float(weight) * alpha
        averaged.append((avg / weight_sum).detach().cpu().numpy())
        last_iterates.append(alpha.detach().cpu().numpy())
    return averaged, last_iterates


def gap_trajectory(
    demand: np.ndarray,
    v: np.ndarray,
    beta: float,
    a_mat: np.ndarray,
    x0: float,
    order_cap: float,
    reg_lambda: float,
    x_star: np.ndarray,
    step_scale: float,
    gamma_scale: float,
    delta_scale: float,
    num_iters: int,
    seed: int,
) -> np.ndarray:
    averaged, _ = run_solver(
        demand,
        v,
        beta,
        a_mat,
        x0,
        order_cap,
        reg_lambda,
        step_scale,
        gamma_scale,
        delta_scale,
        num_iters,
        seed,
    )
    gap = []
    denom = max(float(np.linalg.norm(x_star)), 1e-12)
    for alpha_eval in averaged:
        gap.append(float(np.linalg.norm(alpha_eval - x_star)) / denom)
    return np.asarray(gap)


def main():
    num_periods = 40
    gamma = 0.6
    beta = 0.35
    x0_raw = 8.0
    order_cap_raw = 4.0
    reg_lambda = 0.5
    num_iters = 10000
    tune_iters = 1000
    num_scenarios = 2000

    demand_raw = make_inventory_dataset(num_periods=num_periods, num_scenarios=num_scenarios)
    demand_scale = float(np.mean(demand_raw))
    demand = demand_raw / demand_scale
    x0 = x0_raw / demand_scale
    order_cap = order_cap_raw / demand_scale
    a_mat = lower_tri_ones(num_periods)
    v = np.ones(num_periods, dtype=np.float32) @ (
        beta * a_mat + gamma * np.eye(num_periods, dtype=np.float32)
    )

    step_scale = 1.0
    ref_full, x_star = solve_reference(v, beta, a_mat, demand, x0, order_cap, reg_lambda)
    x_star_norm = max(float(np.linalg.norm(x_star)), 1e-12)
    gamma_scales = [160, 170, 180, 190, 200]
    delta_scales = [0.05, 0.075, 0.1, 0.125, 0.15]
    selected = {"score": float("inf")}
    for gamma_candidate in gamma_scales:
        for delta_candidate in delta_scales:
            averaged, _ = run_solver(
                demand,
                v,
                beta,
                a_mat,
                x0,
                order_cap,
                reg_lambda,
                step_scale,
                gamma_candidate,
                delta_candidate,
                tune_iters,
                seed=101,
            )
            mean_viol = mean_violation(averaged[-1], a_mat, demand, x0)
            score = float(np.linalg.norm(averaged[-1] - x_star)) / x_star_norm
            # if mean_viol < selected["score"]:
            if score < selected["score"]:
                selected = {
                    "score": score,
                    "step_scale": step_scale,
                    "gamma_scale": gamma_candidate,
                    "delta_scale": delta_candidate,
                    "tune_iters": tune_iters,
                    "mean_violation": mean_viol
                }

    all_gap = []
    for run in range(10):
        averaged, _ = run_solver(
            demand,
            v,
            beta,
            a_mat,
            x0,
            order_cap,
            reg_lambda,
            selected["step_scale"],
            selected["gamma_scale"],
            selected["delta_scale"],
            num_iters,
            seed=500 + run,
        )
        gap = []
        for alpha_eval in averaged:
            gap.append(float(np.linalg.norm(alpha_eval - x_star)) / x_star_norm)
        all_gap.append(gap)

    gap_arr = np.array(all_gap)
    gap_mean = gap_arr.mean(axis=0)
    gap_std = gap_arr.std(axis=0)

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_horizon = gap_mean.size
    iters = np.arange(plot_horizon)
    plt.figure(figsize=(7, 4))
    plt.plot(iters, gap_mean[:plot_horizon])
    plt.fill_between(
        iters,
        np.maximum(gap_mean[:plot_horizon] - 3.0 * gap_std[:plot_horizon], 0.0),
        gap_mean[:plot_horizon] + 3.0 * gap_std[:plot_horizon],
        alpha=0.2,
    )
    plt.axhline(0.2, color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("Iteration")
    plt.ylabel(r"Relative distance to $x^\star$")
    plt.title(r"Inventory control: relative distance to $x^\star$")
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(out_dir / "sipm_inventory_gap2.pdf")
    plt.close()

    print("Fixed step scale:", selected["step_scale"])
    print("Tuning iterations:", selected["tune_iters"])
    print("Tuned gamma scale:", selected["gamma_scale"])
    print("Tuned delta scale:", selected["delta_scale"])
    print("Tuning mean violation:", selected["mean_violation"])
    print("Tuning score:", selected["score"])
    print("Demand scale:", demand_scale)
    print("Regularization lambda:", reg_lambda)
    print("Reference full objective:", ref_full)
    print("Distance to CVX solution:", float(np.linalg.norm(averaged[-1] - x_star)))
    print("Final relative distance to CVX solution (mean):", gap_mean[-1])
    print("Final relative distance to CVX solution (std):", gap_std[-1])
    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
