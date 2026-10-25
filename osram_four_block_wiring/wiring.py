"""Independent connection functions. No branch modifies the parent Memory."""
import torch


def neumiss_prediction(reader, block, frame):
    return reader.decode_delta(frame, block(frame))


def signature_prediction(reader, block, frame):
    return reader.decode_delta(frame, block(frame))


def ridge_prediction(block, frame, bank, conversation_ids, rate_index, parent_id):
    # Flatten valid rows in B,T order, matching existing task metric utilities.
    valid = frame.valid.T
    hidden = frame.hidden.transpose(0, 1)[valid]
    a = frame.availability.transpose(0, 1)[valid]
    bits = torch.tensor([4, 2, 1],device=a.device,dtype=torch.long)
    patterns = (a.long() * bits).sum(-1)
    ids = [str(conversation_ids[b]) for b, t in valid.nonzero().tolist()]
    delta = block(hidden, bank, ids, rate_index, patterns, parent_id)
    output = frame.prediction.transpose(0, 1).clone()
    output[valid] = output[valid] + delta
    return output.transpose(0, 1)


def csdi_noise_loss(reader, denoiser, projection, frame, complete_train, generator, split):
    if split != 'train':
        raise ValueError('Complete features are permitted only for TRAIN denoising targets')
    parent = reader.parent
    with torch.no_grad():
        parts = torch.split(complete_train, parent.observed_set.dimensions, -1)
        targets = []
        for name, part in zip(('audio','text','visual'), parts):
            target = part.new_zeros(*part.shape[:2], 256)
            target[frame.valid] = parent.observed_set.projectors[name](part[frame.valid])
            targets.append(target)
        target = projection(torch.stack(targets, -2))
        observed = projection(frame.latents, frame.availability)
    return denoiser.noise_loss(observed, target, frame.availability,
                              frame.context, frame.valid, generator)


def csdi_prediction(reader, denoiser, bridge, projection, frame, generator=None,
                    n_samples=4, aggregation='prediction', samples=None):
    if samples is None:
        samples = denoiser.sample(projection(frame.latents, frame.availability),
            frame.availability, frame.context, frame.valid, n_samples, generator)
    if aggregation == 'latent_mean':
        samples = samples.mean(2, keepdim=True)
    elif aggregation != 'prediction':
        raise ValueError('Unknown hypothesis aggregation')
    delta = bridge(frame.local, samples, frame.availability, frame.valid)
    return reader.decode_delta(frame, delta).mean(2)
