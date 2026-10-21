"""Finite MOSEI queue/resource/report contracts; no Torch or GPU required."""
import json
from types import SimpleNamespace

import pytest

from experiments.osram_mosei_nested_gcnet_3seed_20261021 import run, report


def test_exact_five_new_jobs_and_protected_flat66_reference():
    jobs = run.plan_jobs()
    assert {(v['mode'], v['seed']) for v in jobs} == {
        ('nested', 66), ('nested', 67), ('nested', 68), ('flat', 67), ('flat', 68)}
    assert len({v['name'] for v in jobs}) == 5
    assert report.FLAT66_REFERENCE.name == 'flat'
    assert 'osram_mosei_gcnet_router_20261020' in str(report.FLAT66_REFERENCE)


def test_gpu_identity_forbids_bad_card_unknown_uuid_and_duplicates():
    rows = '0, GPU-good0, 22000\n4, GPU-bad4, 32000\n7, GPU-good7, 18000\n'
    assert run.parse_gpus(rows, ['GPU-good0', 'GPU-good7'])['GPU-good7']['index'] == 7
    for uuids in (['GPU-bad4'], ['GPU-unknown'], ['GPU-good0', 'GPU-good0'], ['0']):
        with pytest.raises(ValueError): run.parse_gpus(rows, uuids)
    with pytest.raises(ValueError):
        run.parse_gpus('2, '+run.BAD_UUID+', 30000\n', [run.BAD_UUID])


def test_admission_reserves_unmaterialized_loading_jobs_and_ram():
    # CUDA usage may not yet appear in nvidia-smi during CPU imports/data loading.
    assert run.job_reserve_mib('nested', 3000) == 8192
    assert run.job_reserve_mib('flat', 3000) == 4096
    assert run.can_admit(free_mib=20000, existing_reserved_mib=10240,
                         proposed_mib=5120, ram_gib=64, active_count=2)
    assert not run.can_admit(free_mib=12000, existing_reserved_mib=10240,
                             proposed_mib=5120, ram_gib=100, active_count=2)
    assert not run.can_admit(free_mib=30000, existing_reserved_mib=0,
                             proposed_mib=5120, ram_gib=63.9, active_count=0)
    assert not run.can_admit(free_mib=30000, existing_reserved_mib=0,
                             proposed_mib=5120, ram_gib=100, active_count=4)


def test_worker_command_uses_uuid_independent_paths_and_optional_full_resume(tmp_path):
    args = SimpleNamespace(output=tmp_path, source_commit='a'*40)
    cmd = run.worker_command(args, {'mode': 'nested', 'seed': 68}, 'train', 'GPU-good7')
    assert cmd[cmd.index('--seed')+1] == '68'
    assert cmd[cmd.index('--mode')+1] == 'nested'
    assert cmd[cmd.index('--gpu-uuid')+1] == 'GPU-good7'
    assert cmd[-1] == str(tmp_path/'runs/CMUMOSEI/seed_68/nested')
    assert '--resume' not in cmd
    assert '--resume' in run.worker_command(args, {'mode': 'nested', 'seed': 68}, 'train', 'GPU-good7', resume=True)


def fixture_summary(seed, mode, wf1):
    rates = report.RATES
    return dict(status='complete', epoch=60, seed=seed, mode=mode,
                reference_pair_verified=True,
                identity=dict(dataset_manifest_sha256='data', split='split',
                              seed_protocol={'seed': seed}, seed=seed,
                              mask_hashes={'test': {r:'mask' for r in rates}}),
                initial_backbone_sha256='init'+str(seed),
                config=dict(seed=seed, learning_rate=1e-3,
                    osram_meaningful_block='none' if mode=='flat' else 'nested_gnn_rooted_evidence'),
                source_commit='sha', peak_memory_mib=3000, seconds=100,
                evaluation={r: {'router': {'ACC': wf1+0.2, 'WF1': wf1, 'MSE': 0.5}} for r in rates},
                best_by_rate={r: {'epoch': 25} for r in rates},
                aggregate={'router': {s: {'ACC': wf1+0.2, 'WF1': wf1, 'MSE': 0.5} for s in ('mean', 'high')}})


def put_summary(root, seed, mode, state):
    path = root/'runs/CMUMOSEI'/f'seed_{seed}'/mode/'SUMMARY.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def test_report_excludes_incomplete_and_never_claims_three_seed_mean(tmp_path):
    reference = tmp_path/'reference'; reference.mkdir()
    (reference/'SUMMARY.json').write_text(json.dumps(fixture_summary(66, 'flat', 83)))
    put_summary(tmp_path, 66, 'nested', fixture_summary(66, 'nested', 84))
    put_summary(tmp_path, 67, 'nested', dict(fixture_summary(67, 'nested', 99), status='running', epoch=10))
    result = report.summarize(tmp_path, flat66_reference=reference)
    assert result['three_seed'] == []
    assert len(result['per_rate']) == 16
    assert all(v['seed'] == 66 for v in result['per_rate'])
    assert len(result['paired_aggregate']) == 2
    assert result['paired_aggregate'][0]['delta_WF1_pp'] == 1
    assert 'INTERNAL TEST-ORACLE SCREEN' in (tmp_path/'RESULT.md').read_text()


def test_complete_report_sample_std_and_paired_three_seed_deltas(tmp_path):
    reference = tmp_path/'reference'; reference.mkdir()
    (reference/'SUMMARY.json').write_text(json.dumps(fixture_summary(66, 'flat', 80)))
    for seed, f in [(66, 80), (67, 81), (68, 82)]:
        if seed != 66: put_summary(tmp_path, seed, 'flat', fixture_summary(seed, 'flat', f))
        put_summary(tmp_path, seed, 'nested', fixture_summary(seed, 'nested', f+0.25))
    result = report.summarize(tmp_path, flat66_reference=reference)
    assert len(result['per_rate']) == 48
    flat_mean = next(v for v in result['three_seed'] if v['mode']=='flat' and v['scope']=='mean')
    assert flat_mean['WF1_mean'] == 81
    assert flat_mean['WF1_sample_std'] == 1
    assert len(result['paired_aggregate']) == 6
    assert all(v['delta_WF1_pp'] == 0.25 for v in result['paired_aggregate'])


def test_report_rejects_complete_but_missing_rates(tmp_path):
    reference = tmp_path/'reference'; reference.mkdir()
    state = fixture_summary(66, 'flat', 83)
    del state['evaluation']['0.7']
    (reference/'SUMMARY.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='rate'):
        report.summarize(tmp_path, flat66_reference=reference)


def test_unverified_new_run_cannot_enter_a_paired_table(tmp_path):
    reference = tmp_path/'reference'; reference.mkdir()
    (reference/'SUMMARY.json').write_text(json.dumps(fixture_summary(66, 'flat', 83)))
    state = fixture_summary(66, 'nested', 84)
    state.pop('reference_pair_verified')
    put_summary(tmp_path, 66, 'nested', state)
    with pytest.raises(ValueError, match='verification'):
        report.summarize(tmp_path, flat66_reference=reference)


def test_queue_failure_stops_further_admissions_without_killing_children(tmp_path, monkeypatch):
    commands = []
    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.pid = 100+len(commands)
            self.smoke = command[command.index('--phase')+1] == 'smoke'
            output = __import__('pathlib').Path(command[command.index('--output')+1])
            output.mkdir(parents=True)
            (output/'SUMMARY.json').write_text(json.dumps(dict(
                status='complete' if self.smoke else 'failed', peak_memory_mib=3000)))
        def wait(self): return 0 if self.smoke else 1
        def poll(self): return self.wait()
    monkeypatch.setattr(run.subprocess, 'Popen', Process)
    monkeypatch.setattr(run, 'gpu_snapshot', lambda _: {'GPU-good7': dict(index=7, free_mib=11000)})
    monkeypatch.setattr(run, 'ram_available_gib', lambda: 100)
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(run.time, 'sleep', lambda _: None)
    monkeypatch.setattr(run, 'summarize', lambda _: None)
    monkeypatch.setattr(run.sys, 'argv', ['run', '--output', str(tmp_path/'new'),
        '--gpus', 'GPU-good7', '--source-commit', 'a'*40])
    run.main()
    assert len(commands) == 2  # one successful smoke, one failed train; four remain unstarted
    queue = json.loads((tmp_path/'new/QUEUE.json').read_text())
    assert queue['status'] == 'failed'
    assert queue['pending_count'] == 4


@pytest.mark.parametrize('change', ['lr', 'data', 'split', 'mask', 'initial', 'seed_protocol'])
def test_report_rejects_actual_pair_mismatches(change):
    flat = fixture_summary(66, 'flat', 83)
    nested = fixture_summary(66, 'nested', 84)
    if change == 'lr': nested['config']['learning_rate'] = 1e-4
    elif change == 'initial': nested['initial_backbone_sha256'] = 'different'
    elif change == 'mask': nested['identity']['mask_hashes']['test']['0.7'] = 'different'
    else:
        key = {'data':'dataset_manifest_sha256', 'split':'split', 'seed_protocol':'seed_protocol'}[change]
        nested['identity'][key] = 'different'
    with pytest.raises(ValueError): report.verify_pair(flat, nested)


def test_pair_normalizes_only_four_reviewed_legacy_defaults_and_mask_wrapper():
    flat = fixture_summary(66, 'flat', 83)
    nested = fixture_summary(66, 'nested', 84)
    flat['identity']['mask_hashes'] = flat['identity']['mask_hashes']['test']
    nested['config'].update(osram_source_role=False, osram_source_role_mode='residual',
        osram_source_relation=False, osram_source_relation_layout='source-mean')
    report.verify_pair(flat, nested)
    nested['config']['unreviewed_new_default'] = False
    with pytest.raises(ValueError): report.verify_pair(flat, nested)
