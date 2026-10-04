"""Explicit one-off GPU4 diagnostic authorization; does not enable training."""
import datetime
import json
import os
import subprocess
import time

UUID = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'


def emit(stage, **fields):
    print(json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          **fields)), flush=True)


def health():
    return subprocess.check_output(['nvidia-smi', '-i', UUID,
        '--query-gpu=index,uuid,temperature.gpu,memory.free,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUID
    before = health()
    assert before.split(',')[0].strip() == '4'
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True, timeout=10)
    assert UUID not in apps, 'Target has an existing compute process; refuse overlap'
    emit('before', health=before)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(66)
    torch.cuda.init()
    assert torch.cuda.device_count() == 1
    emit('cuda_initialized', torch=torch.__version__, device=torch.cuda.get_device_name(0))
    a = torch.randn(256, 256, dtype=torch.float64)
    b = torch.randn(256, 256, dtype=torch.float64)
    reference = a @ b
    for dtype, atol, rtol in [(torch.float64, 1e-10, 1e-10), (torch.float32, 2e-4, 2e-4)]:
        actual = (a.to(device='cuda', dtype=dtype) @ b.to(device='cuda', dtype=dtype)).cpu().double()
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        emit('matmul_pass', dtype=str(dtype), max_abs_error=float((actual-reference).abs().max()))
    free, _ = torch.cuda.mem_get_info()
    assert free > 12 * 1024**3, 'Insufficient diagnostic memory margin'
    chunks = [torch.empty(256 * 1024**2, dtype=torch.uint8, device='cuda') for _ in range(32)]
    for pattern in (0, 85, 170, 255):
        for chunk in chunks:
            chunk.fill_(pattern)
        torch.cuda.synchronize()
        for i, chunk in enumerate(chunks):
            assert bool(torch.all(chunk == pattern)), f'Memory mismatch chunk={i} pattern={pattern}'
        emit('memory_pattern_pass', allocated_gib=8, pattern=pattern)
    del chunks, chunk
    torch.cuda.empty_cache()
    x = torch.randn(64, 128)
    w = torch.randn(128, 64)
    gradients = []
    for device in ('cpu', 'cuda'):
        xx = x.to(device).clone().requires_grad_()
        ww = w.to(device).clone().requires_grad_()
        loss = (xx @ ww).square().mean()
        loss.backward()
        gradients.append((xx.grad.cpu(), ww.grad.cpu(), float(loss.detach())))
    for cpu, gpu in zip(gradients[0][:2], gradients[1][:2]):
        torch.testing.assert_close(cpu, gpu, atol=2e-5, rtol=2e-4)
    emit('backward_pass', cpu_loss=gradients[0][2], gpu_loss=gradients[1][2])
    model = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 1)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xx = torch.randn(256, 512, device='cuda')
    target = torch.randn(256, 1, device='cuda')
    started = time.monotonic()
    steps = 0
    while time.monotonic() - started < 20:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(xx)-target).square().mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        optimizer.step()
        torch.cuda.synchronize()
        steps += 1
    after = health()
    assert int(after.split(',')[2]) < 85, 'Temperature above diagnostic ceiling'
    assert before.split(',')[-2:] == after.split(',')[-2:], 'Uncorrected ECC count increased'
    emit('passed', optimizer_steps=steps, health=after,
         limits='Short synthetic test and 8 GiB only; not full VRAM coverage or long training certification; training ban unchanged')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        emit('failed', error=repr(error))
        raise
