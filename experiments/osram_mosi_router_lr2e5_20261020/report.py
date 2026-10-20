"""Reuse strict selected-prediction verification for the specified LR only."""
from experiments.osram_mosi_router_lr1e5_20261020.report import write_report as common_report
from .worker import BACKBONE_LR


def write_report(output, queue):
    return common_report(output, queue, backbone_lr=BACKBONE_LR)
