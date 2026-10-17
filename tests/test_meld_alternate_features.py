from copy import deepcopy
import numpy as np
import pytest


def fixture():
    speakers = {k: np.eye(9)[[0, 1]] for k in range(3)}
    labels = {k: [0, 4] for k in range(3)}
    old = {k: np.ones((2, 5), dtype=np.float32) for k in range(3)}
    text = {k: ['Hi', 'Great!'] for k in range(3)}
    reference = [speakers, labels, deepcopy(labels), *([old] * 8), old, old, text,
                 [0], [2], [1]]
    rows = {s: [dict(Dialogue_ID='0', Utterance_ID=str(t), Emotion=label,
                        Speaker=name, Utterance=line)
                for t, label, name, line in [(0, 'neutral', 'A', 'Hi'), (1, 'joy', 'B', 'Great!')]]
            for s in ('train', 'dev', 'test')}
    records = {s: {(0, t): dict(audio=np.full(3, t + 1.), text=np.full(4, t + 2.),
                    visual=np.full(2, t + 3.), label=label, speaker=name)
                   for t, label, name in [(1, 'joy', 'B'), (0, 'neutral', 'A')]}
               for s in rows}
    return reference, rows, records


def test_conversion_preserves_ids_labels_speakers_splits_and_canonical_order():
    from experiments.osram_meld_alternate_features_20261017.prepare import convert
    reference, rows, records = fixture()
    out, audit = convert(reference, rows, records)
    for i in (0, 1, 2, 13, 14, 15, 16):
        assert out[i] is reference[i]
    np.testing.assert_array_equal(out[11][0], [[1.] * 3, [2.] * 3])
    assert out[3][0].shape == (2, 4)
    assert out[12][0].shape == (2, 2)
    assert audit['utterances'] == {'train': 2, 'dev': 2, 'test': 2}
    assert audit['dimensions'] == {'audio': 3, 'text': 4, 'visual': 2}


def test_missing_ids_labels_speakers_or_nonfinite_values_fail_closed():
    from experiments.osram_meld_alternate_features_20261017.prepare import convert
    for mutation in ('missing', 'label', 'speaker', 'nan'):
        reference, rows, records = fixture()
        if mutation == 'missing':
            del records['train'][(0, 1)]
        elif mutation == 'nan':
            records['train'][(0, 1)]['audio'][0] = np.nan
        else:
            records['train'][(0, 1)][mutation] = 'wrong'
        with pytest.raises(ValueError):
            convert(reference, rows, records)


def test_mmalign_uses_fixed_temporal_means_and_validated_embedding_lookup():
    from experiments.osram_meld_alternate_features_20261017.prepare import mmalign_records
    embedding = np.arange(24, dtype=np.float32).reshape(6, 4)
    bundle = {'0_1': dict(token_ids=np.array([1., 3.]), label=4,
                         audio_features=np.array([[1., 3.], [3., 5.]]),
                         video_features=np.array([[2.], [4.]]))}
    item = mmalign_records(bundle, embedding)[(0, 1)]
    np.testing.assert_array_equal(item['text'], embedding[[1, 3]].mean(0))
    np.testing.assert_array_equal(item['audio'], [2., 4.])
    np.testing.assert_array_equal(item['visual'], [3.])
    bundle['0_1']['token_ids'][0] = 1.5
    with pytest.raises(ValueError):
        mmalign_records(bundle, embedding)


def test_magtkd_uses_author_distilled_streams_and_decodes_labels():
    from experiments.osram_meld_alternate_features_20261017.prepare import magtkd_records
    bundle = dict(vids=[0], dia2utt={0: [2]}, labels={0: [4]}, speakers={0: [7]},
                  text={0: [[1., 2.]]}, audio={0: [[99.]]}, video={0: [[99.]]},
                  audio_kd={0: [[3., 4.]]}, video_kd={0: [[5., 6.]]})
    row = magtkd_records(bundle, {'Rachel': 7})[(0, 2)]
    assert row['label'] == 'neutral' and row['source_speaker'] == 'Rachel'
    np.testing.assert_array_equal(row['audio'], [3., 4.])


def test_upstream_speaker_disagreement_is_audited_without_replacing_canonical_selector():
    from experiments.osram_meld_alternate_features_20261017.prepare import convert
    reference, rows, records = fixture()
    records['train'][(0, 1)]['source_speaker'] = 'Other source name'
    out, audit = convert(reference, rows, records)
    assert out[0] is reference[0]
    assert audit['upstream_speaker_disagreements'] == [dict(
        split='train', dialogue=0, utterance=1, upstream='Other source name', canonical='B')]
