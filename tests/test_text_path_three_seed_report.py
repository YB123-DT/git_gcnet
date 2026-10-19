import json
from pathlib import Path

import numpy as np
import pytest

from experiments.osram_text_path_router_3seed_20261019.report import CORE, write_report


def fixture(tmp_path):
    output, old = tmp_path/'new',tmp_path/'old'
    output.mkdir()
    for dataset in ('MELD','IEMOCAPSix'):
        for seed in (66,67,68):
            for mode in ('flat','joint'):
                root=(old/dataset if seed==66 else output/'runs'/dataset/f'seed_{seed}')/mode
                (root/'evaluation').mkdir(parents=True)
                score=dict(ACC=float(seed),UA=30.,WF1=50.,CE=1.,N=1)
                summary=dict(status='complete',phase='train',seed=seed,
                    identity=dict(config={'seed':seed},source_hashes={k:'same' for k in CORE}),
                    initial_backbone_sha256=str(seed),validation_mask_hashes={'0.0':'same'},
                    aggregate={'router':dict(mean=score,high=score)},evaluation={'0.0':{'router':score}})
                (root/'SUMMARY.json').write_text(json.dumps(summary))
                (root/'history.json').write_text(json.dumps([{'mask_trace_sha256':str(seed)}]))
                np.savez(root/'evaluation/random_0p0.npz',labels=np.array([1]),speakers=np.array([0]),
                         conversation_ids=np.array(['test']),utterance_indices=np.array([0]),
                         availability=np.array([[1,1,1]]))
    return output,dict(seed66_root=str(old))


def test_reuses_seed66_and_averages_seed_scores_not_utterance_counts(tmp_path):
    output,queue=fixture(tmp_path)
    write_report(output,queue)
    summary=json.loads((output/'SUMMARY.json').read_text())
    assert summary['aggregate']['MELD']['joint']['mean']['ACC']==dict(mean=67.,std=1.)
    assert set(summary['runs']['IEMOCAPSix'])=={'66','67','68'}
    assert 'not per-rate Test-best' in (output/'RESULT.md').read_text()


def test_refuses_mixing_different_algorithm_source(tmp_path):
    output,queue=fixture(tmp_path)
    path=output/'runs/MELD/seed_68/joint/SUMMARY.json'
    summary=json.loads(path.read_text());summary['identity']['source_hashes'][CORE[0]]='changed'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='code changed'):
        write_report(output,queue)
