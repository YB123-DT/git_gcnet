"""One constant2e-5 MOSI150 Flat/Router pair, seed66 only."""
from experiments.osram_mosi_router_lr1e5_20261020 import run as common
from .report import write_report

MODULE = 'experiments.osram_mosi_router_lr2e5_20261020.worker'


def plan_jobs(output, source_commit, gpus):
    return common.plan_jobs(output, source_commit, gpus, module=MODULE)


def main():
    common.main(module=MODULE, report=write_report)


if __name__ == '__main__':
    main()
