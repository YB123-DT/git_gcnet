import pytest

from experiments.osram_meld_flat_width_20261014.run import settings_for_width


def source_settings():
    return dict(dataset='MELD', seed=66, epochs=50, batch_size=32,
                learning_rate=.001, missing_mask_protocol='gcnet-random-mask-v1',
                train_rate_mode='cyclic', train_missing_rates=[i / 10 for i in range(8)],
                backbone_type='osram', training_objective='emotion-only',
                osram_bidirectional=False, osram_readout_fusion='flat',
                osram_output_dim=1600, osram_adapter_hidden_dim=0,
                osram_meaningful_block='none', checkpoint_selection='test-oracle-per-rate')


def test_only_adapter_width_changes_and_source_mask_protocol_is_preserved():
    source = source_settings()
    for width in (512, 1024, 2048):
        result = settings_for_width(source, width)
        assert {k for k in result if result[k] != source[k]} == {'osram_adapter_hidden_dim'}
        assert result['train_missing_rates'] == source['train_missing_rates']
        assert result['osram_output_dim'] == 1600
    assert source['osram_adapter_hidden_dim'] == 0


def test_legacy_or_altered_missing_protocol_is_rejected():
    for protocol in ('gcnet-legacy-fixed-v1', 'conversation-mask-v2'):
        source = source_settings()
        source['missing_mask_protocol'] = protocol
        with pytest.raises(ValueError, match='missing_mask_protocol'):
            settings_for_width(source, 512)
