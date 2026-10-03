"""Same-checkpoint, single-memory-scan Flat readout intervention (no training)."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def unpack_diagnostics(diagnostics, valid, heads):
    import numpy as np
    selected = valid.detach().cpu().numpy().astype(bool)
    result = np.full((*selected.shape, 3, 3), np.nan)
    for m, name in enumerate(('audio', 'text', 'visual')):
        for j, metric in enumerate(('rho', 'eta', 'cosine')):
            values = np.asarray(diagnostics[name][metric]).reshape(int(selected.sum()), heads)
            result[..., m, j][selected] = values.mean(axis=-1)
    return result


def observables(base, gap, availability, forward_dim, diagnostics, local_prediction):
    import numpy as np
    import torch
    base = base[:forward_dim]
    active = ~availability.bool()
    gap = torch.where(active[:, None], gap[:, :forward_dim], torch.zeros_like(gap[:, :forward_dim]))
    bn, gn = float(base.norm()), gap.norm(dim=-1)
    result = dict(obs_local_margin=abs(float(local_prediction)), obs_base_norm=bn,
                  obs_gap_base_ratio=float(gn.sum()) / bn if bn > 1e-8 else float('nan'))
    cosines, ds = [], []
    for m, name in enumerate('ATV'):
        result[f'obs_gap_{name}_norm'] = float(gn[m])
        defined = bool(active[m]) and bn > 1e-8 and float(gn[m]) > 1e-8
        cosine = float(torch.nn.functional.cosine_similarity(base, gap[m], dim=0)) if defined else float('nan')
        result[f'obs_base_gap_{name}_cos'] = cosine
        if defined:
            cosines.append(cosine)
        if bool(active[m]):
            ds.append(diagnostics[m])
        for j, metric in enumerate(('rho', 'eta', 'cos')):
            result[f'obs_query_{metric}_{name}'] = float(diagnostics[m, j]) if bool(active[m]) else float('nan')
    result['obs_base_gap_cos_mean'] = float(np.mean(cosines)) if cosines else float('nan')
    for j, metric in enumerate(('rho', 'eta', 'cos')):
        result[f'obs_query_{metric}_mean'] = float(np.asarray(ds)[:, j].mean()) if ds else float('nan')
    return result


def replay_readouts(model, emotion_input, skip, latent_dim, context_dim):
    outputs = []
    for end in (latent_dim, latent_dim + context_dim, emotion_input.shape[-1]):
        x = emotion_input.clone()
        x[..., end:] = 0
        outputs.append(model.smax_fc(model.osram.emotion_norm(skip + model.osram.emotion_adapter(x))))
    return outputs


def mask_head_input(emotion_input, latent_dim, context_dim, value_dim, head, evidence):
    """Zero one forward read slice; retain Local, all other heads and zero half."""
    if evidence not in ('base', 'gap') or head < 0 or (head+1)*value_dim > context_dim // 2:
        raise ValueError('Invalid forward-head intervention')
    result = emotion_input.clone()
    for slot in ((0,) if evidence == 'base' else (1, 2, 3)):
        start = latent_dim + slot * context_dim + head * value_dim
        result[..., start:start+value_dim] = 0
    return result


def query_observables(queries, base, gap, availability, valid, diagnostics, heads, value_dim):
    """Describe existing tensors; raw query comparisons do not undo residual addressing."""
    import numpy as np
    import torch
    valid_np=valid.detach().cpu().numpy().astype(bool)
    raw=np.full((*valid_np.shape,3,3,heads),np.nan)
    for m,name in enumerate(('audio','text','visual')):
        for j,metric in enumerate(('rho','eta','cosine')):
            raw[...,m,j,:][valid_np]=np.asarray(diagnostics[name][metric]).reshape(-1,heads)
    q=queries.detach().cpu()
    b=base.detach().cpu().reshape(*valid.shape,heads,value_dim)
    g=gap.detach().cpu().reshape(*valid.shape,3,heads,value_dim)
    a=availability.detach().cpu().bool()
    def cosine(x,y):
        if float(x.norm())<=1e-8 or float(y.norm())<=1e-8:
            return float('nan')
        return float(torch.nn.functional.cosine_similarity(x,y,dim=-1))
    rows=[];index=0
    for batch in range(valid.shape[1]):
        for time in range(valid.shape[0]):
            if not valid_np[time,batch]:
                continue
            for m,name in enumerate('ATV'):
                if bool(a[time,batch,m]):
                    continue
                for head in range(heads):
                    qb,qg=q[time,batch,0,head],q[time,batch,m+1,head]
                    br,gr=b[time,batch,head],g[time,batch,m,head]
                    rho=float(raw[time,batch,m,0,head])
                    residual_cosine=float(raw[time,batch,m,2,head])
                    if float(qg.norm())<=1e-8 or rho*float(qg.norm())<=1e-8:
                        residual_cosine=float('nan')
                    rows.append(dict(artifact_row=index,modality=name,head=head,
                        cos_base_gap_query=cosine(qb,qg),
                        cos_gap_residual_query=residual_cosine,
                        cos_base_gap_read=cosine(br,gr),rho=rho,
                        eta=float(raw[time,batch,m,1,head]),norm_q_base=float(qb.norm()),
                        norm_q_gap=float(qg.norm()),norm_base_read=float(br.norm()),norm_gap_read=float(gr.norm())))
            index+=1
    return rows


class Capture:
    """Read-only hooks/profile capture; replay never calls encoder, queries or scan."""
    def __init__(self, model, head_ablation=False, query_audit=False):
        self.model = model
        self.batches = []
        self.scans = 0
        self.replaying = False
        self.handles = []
        self.head_ablation = head_ablation
        self.head_batches = []
        self.query_audit = query_audit
        self.query_batches = []

    def profile(self, frame, event, arg):
        if event == 'return' and frame.f_code is self.model.osram._scan.__func__.__code__:
            state = frame.f_locals
            assert not state['reverse']
            self.scan_diagnostics = unpack_diagnostics(state['diagnostics'], state['valid'], self.model.osram.num_heads)
            if self.query_audit:
                rows=query_observables(state['queries'],state['base'],state['gap'],state['availability'],
                    state['valid'],state['diagnostics'],self.model.osram.num_heads,self.model.osram.value_dim)
                self.query_batches.extend(dict(row,artifact_row=row['artifact_row']+len(self.batches)) for row in rows)
            self.scans += 1

    def pre(self, module, args):
        self.availability, self.umask = args[1], args[3]
        self.before_scans = self.scans

    def adapter(self, module, args):
        if not self.replaying:
            self.emotion_input = args[0].detach().clone()

    def skip(self, module, args, output):
        self.skip_output = output.detach().clone()

    def post(self, module, args, output):
        import numpy as np
        import torch
        from experiments.osram_current_history_relation_20261003.residual_off import verify_replay
        assert self.scans == self.before_scans + 1
        latent = self.model.osram.latent_dim
        context = (self.emotion_input.shape[-1] - latent) // 4
        base = self.emotion_input[..., latent:latent+context]
        gap = self.emotion_input[..., latent+context:].reshape(*base.shape[:2], 3, context)
        valid = self.umask.T.bool()
        inactive = self.availability.bool() | ~valid.unsqueeze(-1)
        assert torch.equal(gap[inactive], torch.zeros_like(gap[inactive]))
        self.replaying = True
        try:
            predictions = replay_readouts(module, self.emotion_input, self.skip_output, latent, context)
        finally:
            self.replaying = False
        selected = self.umask.bool()
        def flatten(tensor):
            return tensor.squeeze(-1).T[selected].detach().cpu().numpy()
        pl, pb, pf = [flatten(p) for p in predictions]
        verify_replay(pf, flatten(output[0]))
        if self.head_ablation:
            self.replaying = True
            try:
                offset = len(self.batches)
                for evidence in ('base', 'gap'):
                    for head in range(module.osram.num_heads):
                        modified = mask_head_input(self.emotion_input, latent, context,
                                                   module.osram.value_dim, head, evidence)
                        masked = flatten(module.smax_fc(module.osram.emotion_norm(
                            self.skip_output + module.osram.emotion_adapter(modified))))
                        self.head_batches.extend(dict(artifact_row=offset+i, intervention=evidence,
                            head=head, pred_masked=float(value)) for i,value in enumerate(masked))
            finally:
                self.replaying = False
        first = valid & (valid.long().cumsum(0) == 1)
        full = valid & self.availability.bool().all(-1)
        for mask in (first, full):
            if bool(mask.any()):
                verify_replay(predictions[1][mask].cpu().numpy(), predictions[2][mask].cpu().numpy())
        if bool(first.any()):
            verify_replay(predictions[0][first].cpu().numpy(), predictions[2][first].cpu().numpy())
        rows = []
        index = 0
        for b in range(valid.shape[1]):
            for t in range(valid.shape[0]):
                if not bool(valid[t, b]):
                    continue
                obs = observables(base[t,b], gap[t,b], self.availability[t,b],
                                  self.model.osram.num_heads * self.model.osram.value_dim,
                                  self.scan_diagnostics[t,b], pl[index])
                rows.append(dict(pred_local=float(pl[index]), pred_base=float(pb[index]), pred_full=float(pf[index]),
                                 **obs))
                index += 1
        self.batches.extend(rows)

    def __enter__(self):
        assert sys.getprofile() is None
        self.handles = [self.model.register_forward_pre_hook(self.pre),
                        self.model.osram.emotion_adapter.register_forward_pre_hook(self.adapter),
                        self.model.osram.local_skip.register_forward_hook(self.skip),
                        self.model.register_forward_hook(self.post)]
        sys.setprofile(self.profile)
        return self

    def __exit__(self, *args):
        sys.setprofile(None)
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gpu', default='5')
    parser.add_argument('--head-ablation', action='store_true',
                        help='Also replay 16 classifier-only single-head interventions; never another scan')
    parser.add_argument('--query-audit', action='store_true',help='Capture existing per-head queries/read values/diagnostics')
    parser.add_argument('--reference-dir', type=Path, default=ROOT/'experiments/osram_cfg84_history_scale_20260929/results')
    args = parser.parse_args()
    import numpy as np
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, evaluate_rate
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model, _roots
    from experiments.osram_current_history_relation_20261003.residual_off import verify_replay
    from experiments.osram_paired_history_rho025_20261002.sweep import gpu_check, UUIDS
    from experiments.osram_cfg84_history_query_random_20260928.run import sha, write
    assert args.gpu != '4' and os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu
    assert gpu_check(args.gpu) > 2500
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.source/'config.json').read_text())
    c = TrainConfig(**config)
    assert c.seed == 66 and c.osram_readout_fusion == 'flat' and c.training_objective == 'emotion-only'
    assert not c.osram_relation_block and c.osram_ablation == 'full' and c.osram_emotion_ablation == 'full'
    assert c.mosi_task_mode == 'regression' and c.completion_path == 'none'
    for flag in ('osram_post_grn', 'osram_local_skip_gate', 'osram_history_input_gate',
                 'osram_local_evidence_gate', 'osram_hierarchical_evidence_gate',
                 'osram_history_query_adapter', 'paired_history_views', 'osram_relation_dual_readout'):
        assert not getattr(c, flag, False), flag
    ids_path = ROOT/'experiments/osram_cfg84_history_scale_fine_20260929/prediction_analysis/reconstructed_sample_ids.csv'
    with ids_path.open() as stream:
        ids = sorted((r for r in csv.DictReader(stream) if int(r['seed']) == 66), key=lambda r:int(r['artifact_row']))
    assert [int(r['artifact_row']) for r in ids] == list(range(len(ids)))
    provenance = dict(status='running', training=False, code_commit=args.commit, source=str(args.source),
                      server='biggpu', gpu=args.gpu, gpu_uuid=UUIDS[args.gpu], pid=os.getpid(),
                      config_sha256=sha(args.source/'config.json'), ids_sha256=sha(ids_path),
                      source_sha256={name:sha(ROOT/name) for name in ('gcnet_missing_m3/model.py',
                          'gcnet_missing_m3/osram.py','gcnet_missing_m3/train_gcnet.py',
                          str(Path(__file__).relative_to(ROOT)))},
                      environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version),
                      utterance_index_semantics='Original sample-ID numeric suffix; not a reconstructed zero-based sequence index',
                      head_ablation=args.head_ablation,
                      query_audit=args.query_audit,
                      query_comparison='qB vs raw qG; read G uses residual-addressed qG. Do not interpret raw-query/read mismatch alone as mapping collapse.',
                      intervention='Same full checkpoint; one scan; only Flat history inputs zeroed',
                      protocol='INTERNAL DIAGNOSTIC ONLY; original per-rate Test-oracle BEST')
    write(args.output/'PROVENANCE.json', provenance)
    try:
        torch.set_num_threads(2)
        _, _, loaders, ad, td, vd = get_loaders(audio_root=_roots()[0], text_root=_roots()[1], video_root=_roots()[2],
            num_folder=1, dataset=c.dataset, batch_size=c.batch_size, num_workers=0, seed=c.seed,
            validation_fraction=c.validation_fraction, evaluation_protocol=c.evaluation_protocol)
        model = _build_model(c, (ad,td,vd)).cuda().eval().requires_grad_(False)
        assert not model.osram.bidirectional
        schedules = _schedules(c, 'test')
        all_rows, checks, all_head_rows, all_query_rows = [], [], [], []
        for rate in [i/10 for i in range(8)]:
            checkpoint = args.source/f'best_miss_{str(rate).replace(".", "p")}.pt'
            checkpoint_sha = sha(checkpoint)
            state = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(state['model'], strict=True)
            reference_path = args.reference_dir/f'seed_66_miss_{rate:.1f}_alpha_1.0.npz'
            with np.load(reference_path) as stream:
                reference = {key:stream[key].copy() for key in stream.files}
            with torch.no_grad(), Capture(model, head_ablation=args.head_ablation,query_audit=args.query_audit) as capture:
                metrics, artifacts = evaluate_rate(model, loaders[0], schedules[rate], c.dataset,
                    (ad,td,vd), torch.device('cuda:0'), True, c.mosi_task_mode,
                    c.task_regression_loss, c.task_smooth_l1_beta)
            for key in ('labels','availability'):
                np.testing.assert_array_equal(artifacts[key], reference[key])
            replay_error = verify_replay(artifacts['predictions'], reference['predictions'])
            assert len(capture.batches) == len(ids) == len(artifacts['labels'])
            for i, row in enumerate(capture.batches):
                y = float(artifacts['labels'][i]); uid = ids[i]['utterance_id']
                assert float(ids[i]['label']) == y
                conversation, index = uid.rsplit('_',1)
                bcorrect = (row['pred_base'] > 0) == (y > 0)
                fcorrect = (row['pred_full'] > 0) == (y > 0)
                category = 'neutral_excluded' if y == 0 else ('both_correct' if bcorrect and fcorrect else
                           'both_wrong' if not bcorrect and not fcorrect else 'rescue' if fcorrect else 'harm')
                all_rows.append(dict(seed=66,rate=rate,artifact_row=i,utterance_id=uid,conversation_id=conversation,
                    utterance_index=int(index),label=y,availability=''.join(m for m,a in zip('ATV',artifacts['availability'][i]) if a),
                    delta_gap=(y-row['pred_base'])**2-(y-row['pred_full'])**2,polarity_category=category,**row))
            rate_rows = all_rows[-len(ids):]
            for intervention in capture.head_batches:
                original = rate_rows[intervention['artifact_row']]
                all_head_rows.append(dict(seed=66,rate=rate,utterance_id=original['utterance_id'],
                    label=original['label'],availability=original['availability'],
                    pred_full=original['pred_full'],**intervention))
            for query in capture.query_batches:
                original=rate_rows[query['artifact_row']]
                all_query_rows.append(dict(seed=66,rate=rate,utterance_id=original['utterance_id'],
                    label=original['label'],availability=original['availability'],**query))
            for key,value in model.state_dict().items():
                assert torch.equal(value.cpu(),state['model'][key]), key
            assert sha(checkpoint) == checkpoint_sha
            checks.append(dict(rate=rate,epoch=state.get('epoch'),checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
                reference_sha256=sha(reference_path),full_replay_max_abs_error=replay_error,exact_polarity=True,
                scans=capture.scans,expected_batches=len(loaders[0]),metrics=metrics))
            assert capture.scans == len(loaders[0])
            write(args.output/'checks.json',checks)
            print(f'rate={rate:.1f} rows={len(capture.batches)} scans={capture.scans} full_wf1={100*metrics["weighted_f1"]:.6f}',flush=True)
        with (args.output/'utterances.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
        if args.head_ablation:
            assert len(all_head_rows) == len(all_rows) * 2 * model.osram.num_heads
            with (args.output/'head_predictions.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(all_head_rows[0]));writer.writeheader();writer.writerows(all_head_rows)
        if args.query_audit:
            with (args.output/'query_observables.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(all_query_rows[0]));writer.writeheader();writer.writerows(all_query_rows)
        provenance.update(status='complete',rows=len(all_rows),scans=sum(r['scans'] for r in checks))
    except BaseException as error:
        provenance.update(status='failed',error=repr(error));raise
    finally:
        write(args.output/'PROVENANCE.json',provenance)


if __name__ == '__main__':
    main()
