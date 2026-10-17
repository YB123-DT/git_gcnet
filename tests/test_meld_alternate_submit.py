def test_command_keeps_remote_environment_data_hash_and_single_seed():
    from experiments.osram_meld_alternate_features_20261017.submit import command
    text = command('GPU-healthy', '59eb', '/run', smoke=False)
    assert text[0] == '/data2/yb/conda_envs/cmerc-repro/bin/python'
    assert text[text.index('--data-sha256') + 1] == '59eb'
    assert '--smoke' not in text
    assert command('GPU-healthy', '59eb', '/smoke', smoke=True)[-1] == '--smoke'


def test_gpu_admission_rejects_forbidden_card_and_counts_other_processes():
    from experiments.osram_meld_alternate_features_20261017.submit import choose_gpu
    inventory = '0, GPU-zero, 30000\n3, GPU-three, 20000\n4, GPU-bad, 32000\n'
    busy = '\n'.join(f'GPU-three, {i}' for i in range(4))
    assert choose_gpu(inventory, busy) == 'GPU-zero'
    assert choose_gpu('4, GPU-bad, 32000\n', '') is None
    assert choose_gpu('3, GPU-three, 2000\n', '') is None
