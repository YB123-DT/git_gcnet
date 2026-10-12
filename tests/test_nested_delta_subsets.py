import torch
from experiments.osram_nested_delta_off_missing_text_20261010.readout import (
    disable_correction_slot,
    scale_correction_slot,
)

def test_multiple_slots_restore_only_selected_evidence():
    raw=(torch.zeros(1,1,2),torch.zeros(1,1,4),torch.zeros(1,1,3,4))
    corrected=tuple(x+1 for x in raw)
    out=disable_correction_slot(raw,corrected,'Local+Gap-T+Gap-V')
    assert not out[0].count_nonzero()
    assert torch.equal(out[1],corrected[1])
    assert torch.equal(out[2][:,:,0],corrected[2][:,:,0])
    assert not out[2][:,:,1:].count_nonzero()
    out=disable_correction_slot(raw,corrected,'Local+Base+Gap-A+Gap-T+Gap-V')
    assert all(torch.equal(x,y) for x,y in zip(out,raw))
    assert all(torch.equal(x,torch.ones_like(x)) for x in corrected)


def test_scale_one_correction_slot_leaves_other_slots_unchanged():
    raw=(torch.full((1,1,2), 10.),torch.full((1,1,4), 20.),
         torch.full((1,1,3,4), 30.))
    corrected=(raw[0]+1,raw[1]+2,raw[2]+3)
    expected={
        'Local': (raw[0]+.5, corrected[1], corrected[2]),
        'Base': (corrected[0], raw[1]+1, corrected[2]),
        'Gap-A': (corrected[0], corrected[1], torch.cat((raw[2][:,:,:1]+1.5, corrected[2][:,:,1:]),2)),
        'Gap-T': (corrected[0], corrected[1], torch.cat((corrected[2][:,:,:1], raw[2][:,:,1:2]+1.5, corrected[2][:,:,2:]),2)),
        'Gap-V': (corrected[0], corrected[1], torch.cat((corrected[2][:,:,:2], raw[2][:,:,2:]+1.5),2)),
    }
    for slot, wanted in expected.items():
        actual=scale_correction_slot(raw,corrected,slot,.5)
        assert all(torch.equal(x,y) for x,y in zip(actual,wanted))
    assert all(torch.equal(x,y) for x,y in zip(scale_correction_slot(raw,corrected,'Local',1.),corrected))
    assert all(torch.equal(x,y) for x,y in zip(scale_correction_slot(raw,corrected,'Base',0.),
                                                disable_correction_slot(raw,corrected,'Base')))


def test_scale_correction_slot_rejects_invalid_inputs_without_mutation():
    raw=(torch.zeros(1,1,2),torch.zeros(1,1,4),torch.zeros(1,1,3,4))
    corrected=tuple(x+1 for x in raw)
    for slot, scale in [('Unknown',1.),('Local',float('nan')),('Base',float('inf'))]:
        try:
            scale_correction_slot(raw,corrected,slot,scale)
        except ValueError:
            pass
        else:
            raise AssertionError((slot,scale))
    assert all(torch.equal(x,torch.ones_like(x)) for x in corrected)
