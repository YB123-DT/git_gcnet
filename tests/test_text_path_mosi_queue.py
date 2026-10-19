import pytest


def test_mosi_queue_has_only_seed66_flat_and_router_commands(tmp_path):
    from experiments.osram_text_path_router_joint_mosi_20261019.orchestrate import command
    for mode in ('flat','joint'):
        args=command(mode,'train','source',tmp_path/mode,'GPU-valid')
        assert args[args.index('--mode')+1]==mode
        assert args[args.index('--phase')+1]=='train'
        assert '--source-commit' in args
        assert '--seed' not in args  # fixed seed66, no automatic expansion


def test_mosi_queue_never_allocates_bad_physical_card(monkeypatch):
    from experiments.osram_text_path_router_joint_mosi_20261019 import orchestrate
    monkeypatch.setattr(orchestrate.subprocess,'check_output',lambda *a,**k:
                        '4, GPU-bad, 0, 32768\n')
    with pytest.raises(ValueError,match='forbidden'):
        orchestrate.gpu_info('GPU-bad')
