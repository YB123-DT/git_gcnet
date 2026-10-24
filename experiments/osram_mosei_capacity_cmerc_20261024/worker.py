"""Eight fresh MOSEI capacity arms using the accepted cmerc1e-4 training loop."""
import argparse
from dataclasses import asdict, replace
import sys

from experiments.osram_nested_cmerc_tuning_20261023 import worker as shared

MODES={'f128':(128,'none'),'f384':(384,'none'),'f768':(768,'none'),
    'f256_n64':(256,'nested_gnn_rooted_evidence'),'f256':(256,'none'),
    'f256_n128':(256,'nested_dim128'),'f512_n128':(512,'nested_dim128'),
    'f512_n256':(512,'mosei_capacity_nested256')}


def candidate_config(cfg,mode):
    if mode not in MODES or cfg.osram_adapter_hidden_dim!=0:
        raise ValueError('Only the eight cmerc capacity arms from original Flat are approved')
    shared.candidate_config(cfg,'none',seed=66,fold=1)
    width,method=MODES[mode]
    return replace(cfg,osram_adapter_hidden_dim=width,osram_meaningful_block=method)


def setup(dataset,method,uuid,output,*,seed,fold,mode):
    import torch
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_mosei_v3_capacity_20261022.worker import digest_without
    from experiments.osram_tasr_20261018 import runner_helpers as h
    bundle,large=shared.setup(dataset,'none',uuid,output,seed=seed,fold=fold)
    cfg=candidate_config(bundle['cfg'],mode)
    plain=_build_model(replace(cfg,osram_meaningful_block='none'),bundle['dimensions'],torch.device('cuda:0'))
    model=plain if method=='none' else _build_model(cfg,bundle['dimensions'],torch.device('cuda:0'))
    prefixes=('osram.emotion_adapter.','osram.meaningful_block.')
    if digest_without(large,prefixes)!=digest_without(model,prefixes):
        raise RuntimeError('Capacity changed untouched original initialization')
    initial=h.backbone_digest(plain)
    if digest_without(model,('osram.meaningful_block.',))!=initial:
        raise RuntimeError('Nested changed same-width Flat initialization')
    old_cfg=bundle['cfg'];model.requires_grad_(True)
    bundle.update(model=model,cfg=cfg,initial_backbone_sha256=initial,
        config_reference_delta={k:v for k,v in asdict(cfg).items() if v!=asdict(old_cfg)[k]})
    return bundle,plain


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',choices=tuple(MODES),required=True)
    args,remaining=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    method=MODES[args.mode][1]
    def validate(dataset,actual_method,seed,fold):
        if (dataset,actual_method,seed,fold)!=('CMUMOSEI',method,66,1):
            raise ValueError('MOSI, other seeds/folds/methods are outside this eight-arm task')
    def prepare(*a,**kw):return setup(*a,**kw,mode=args.mode)
    shared.main(['--dataset','CMUMOSEI','--method',method,'--fold','1']+remaining,
        setup_function=prepare,scope_validator=validate,method_choices=(method,),
        additional_source_files=('experiments/osram_mosei_capacity_cmerc_20261024/worker.py',))


if __name__=='__main__':main()
