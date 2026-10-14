"""Explicit training adapter for the retained MELD fixed-mask protocol."""

import torch

from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import LegacyFixedMaskSchedule


class TrainingLegacyFixedMaskSchedule(LegacyFixedMaskSchedule):
    """Reuse the frozen generator exactly, including identical host/guest keys."""

    def generate_batch(self, conversation_ids, umask, epoch):
        if umask.ndim != 2 or len(conversation_ids) != umask.shape[0]:
            raise ValueError("conversation IDs must match [batch, sequence] umask")
        sides = []
        for side in ("host", "guest"):
            masks = [torch.as_tensor(self.generate(
                str(conversation), length=umask.shape[1], side=side, epoch=epoch,
                valid_length=int(umask[index].sum().item())).availability)
                for index, conversation in enumerate(conversation_ids)]
            sides.append(torch.stack(masks, dim=1).to(device=umask.device))
        return tuple(sides)
