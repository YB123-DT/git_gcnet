def test_deployment_uses_fresh_noninteractive_ssh_connections():
    from experiments.osram_text_path_router_3seed_20261019.deploy import ssh_args
    args=ssh_args()
    assert args[:3]==['ssh','-S','none']
    assert 'BatchMode=yes' in args and 'ConnectTimeout=8' in args


def test_remote_deployment_is_idempotent_and_does_not_launch_models_directly():
    from experiments.osram_text_path_router_3seed_20261019.deploy import start_command
    command=start_command()
    assert 'tmux has-session' in command
    assert 'RESOURCE_GATE_STATE.json' in command
    assert 'resource_gate_00b7552.py' in command
    assert 'nvidia-smi' not in command and 'train_gcnet' not in command
    assert 'kill' not in command
