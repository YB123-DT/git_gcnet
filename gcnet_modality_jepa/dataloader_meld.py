"""MELD adapter for the conversation pickle distributed with ECERC."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MELDDataset(Dataset):
    """Expose ECERC MELD features through GCNet's legacy six-stream contract.

    MELD contains one feature vector per utterance, not dyadic host/guest
    streams. Repeating the observed vector in both legacy slots makes the
    downstream binary selector identity-valued while the scalar speaker id
    remains available to the speaker graph.
    """

    def __init__(self, label_path):
        with Path(label_path).open("rb") as stream:
            values = pickle.load(stream, encoding="latin1")
        if len(values) != 17:
            raise ValueError("ECERC MELD pickle must contain 17 fields")
        (speakers, emotion_labels, _sentiment_labels, eroberta1, *_middle,
         audio, visual, _sentences, train_ids, test_ids, valid_ids) = values
        dialogue_ids = set(speakers)
        for name, mapping in (("labels", emotion_labels), ("text", eroberta1),
                              ("audio", audio), ("visual", visual)):
            if set(mapping) != dialogue_ids:
                raise ValueError(f"MELD {name} dialogue ids mismatch")
        split_sets = tuple(map(set, (train_ids, valid_ids, test_ids)))
        if set.union(*split_sets) != dialogue_ids or any(
            split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)
        ):
            raise ValueError("MELD train/valid/test splits are not disjoint and complete")

        self.trainVids, self.valVids, self.testVids = split_sets
        self.vids = list(train_ids) + list(valid_ids) + list(test_ids)
        self.videoLabelsNew = emotion_labels
        self.videoSpeakersNew = {}
        self.videoAudioHost, self.videoTextHost, self.videoVisualHost = {}, {}, {}
        self.videoAudioGuest, self.videoTextGuest, self.videoVisualGuest = {}, {}, {}
        for vid in self.vids:
            arrays = tuple(np.asarray(value[vid]) for value in (audio, eroberta1, visual))
            length = len(emotion_labels[vid])
            if any(array.ndim != 2 or len(array) != length for array in arrays):
                raise ValueError(f"invalid MELD feature shape for dialogue {vid}")
            speaker_matrix = np.asarray(speakers[vid])
            if speaker_matrix.shape != (length, 9):
                raise ValueError(f"invalid MELD speaker shape for dialogue {vid}")
            speaker_ids = speaker_matrix.argmax(axis=1)
            if not np.all(speaker_matrix[np.arange(length), speaker_ids] == 1):
                raise ValueError(f"MELD speakers must be one-hot for dialogue {vid}")
            self.videoSpeakersNew[vid] = speaker_ids.astype(np.float32)
            (self.videoAudioHost[vid], self.videoTextHost[vid],
             self.videoVisualHost[vid]) = tuple(a.astype(np.float32) for a in arrays)
            self.videoAudioGuest[vid] = self.videoAudioHost[vid]
            self.videoTextGuest[vid] = self.videoTextHost[vid]
            self.videoVisualGuest[vid] = self.videoVisualHost[vid]
        first = self.vids[0]
        self.adim = int(self.videoAudioHost[first].shape[-1])
        self.tdim = int(self.videoTextHost[first].shape[-1])
        self.vdim = int(self.videoVisualHost[first].shape[-1])

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        vid = self.vids[index]
        length = len(self.videoLabelsNew[vid])
        return (
            torch.from_numpy(self.videoAudioHost[vid]),
            torch.from_numpy(self.videoTextHost[vid]),
            torch.from_numpy(self.videoVisualHost[vid]),
            torch.from_numpy(self.videoAudioGuest[vid]),
            torch.from_numpy(self.videoTextGuest[vid]),
            torch.from_numpy(self.videoVisualGuest[vid]),
            torch.from_numpy(self.videoSpeakersNew[vid]),
            torch.ones(length),
            torch.as_tensor(self.videoLabelsNew[vid], dtype=torch.long),
            vid,
        )

    def get_featDim(self):
        return self.adim, self.tdim, self.vdim

    def collate_fn(self, data):
        frame = pd.DataFrame(data)
        return [
            pad_sequence(frame[column]) if column <= 5
            else pad_sequence(frame[column], batch_first=True) if column <= 8
            else frame[column].tolist()
            for column in frame
        ]


def load_meld_dataset(label_path):
    return MELDDataset(label_path)
