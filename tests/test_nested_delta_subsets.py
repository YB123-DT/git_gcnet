import torch
from experiments.osram_nested_delta_off_missing_text_20261010.readout import disable_correction_slot

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
