from .sipm import (  # noqa: F401
    SIPMConfig,
    StochasticInexactPenaltyMethod,
    BatchSampler,
    BoxProx,
    DatasetSampler,
    DistributionSampler,
    ElasticNetProx,
    L1Prox,
    L2SquaredProx,
    make_sampler,
    NonNegativityProx,
    SimplexProx,
    SumToOneProx,
    ZeroProx,
    constant_schedule,
    polynomial_schedule,
)


def main() -> None:
    print("sipm module ready")
