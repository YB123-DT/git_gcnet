import importlib.util

import pytest
import torch


def api():
    name='experiments.osram_current_pattern_history_audit_20261022.run'
    assert importlib.util.find_spec(name) is not None, 'fixed-history audit runner missing'
    return __import__(name,fromlist=['select_donor'])


def test_donor_matches_history_length_is_other_conversation_and_label_independent():
    pool={'same':torch.randn(10,4), 'short':torch.randn(2,4), 'good':torch.randn(5,4)}
    assert api().select_donor(pool,'same',6)=='good'
    assert api().select_donor(pool,'same',12) is None
    assert api().select_donor(pool,'same',6)=='good'
    assert api().select_donor(pool,'same',1) is None


def test_donor_view_keeps_recipient_trajectory_and_replaces_history_content_only():
    complete=torch.arange(24,dtype=torch.float).reshape(4,1,6)
    a=torch.tensor([[[1.,0,0]],[[0,1,0]],[[1,0,1]],[[1,1,1]]])
    background={'complete':complete,'availability':a,'umask':torch.ones(1,4),
                'qmask':torch.tensor([[0,1,0,1]]),'labels':torch.arange(4.).reshape(1,4),
                'lengths':[4], 'conversation_ids':['recipient']}
    donor=torch.full((3,6),99.)
    modified=api().replaced_view(background,{'donor':donor},['donor'],(2,2,2))
    assert torch.equal(modified['complete'][:3,0],donor)
    assert torch.equal(modified['complete'][3],complete[3])
    for key in ('availability','umask','qmask','labels'):
        assert torch.equal(modified[key],background[key])
    expanded=torch.repeat_interleave(a,torch.tensor([2,2,2]),dim=-1)
    assert torch.equal(modified['incomplete'],torch.where(expanded.bool(),modified['complete'],torch.zeros_like(complete)))
    assert torch.equal(background['complete'],complete)


def test_current_pattern_view_is_not_a_persistent_history_change():
    a=torch.tensor([[[1.,0,1]],[[0,1,1]],[[1,1,0]]])
    source={'complete':torch.ones(3,1,6),'availability':a,'umask':torch.ones(1,3)}
    changed=api().current_pattern_view(source,(2,2,2),'T')
    assert changed['availability'][:,:,1].eq(1).all()
    assert changed['availability'][:,:,[0,2]].eq(0).all()
    assert torch.equal(source['availability'],a)
    # The synthetic read-only scan uses external factual history states;
    # this view must not be used to generate intervened historical states.


def test_target_selection_never_uses_labels_or_first_utterance():
    view={'conversation_ids':['x','y'],'umask':torch.tensor([[1.,1,1,0],[1,0,0,0]]),
          'labels':torch.zeros(2,4)}
    first=api().choose_targets(view)
    assert len(first)==1 and first[0][0]==0 and first[0][1] in (1,2)
    view['labels'].fill_(99)
    assert first==api().choose_targets(view)
