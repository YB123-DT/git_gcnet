"""Seven-run MELD screen of existing Block2 mechanisms, without structural HPO."""
from copy import deepcopy
import math
from numbers import Real


METHODS = {
    "Flat": {
        "name": "Original Flat1600 + Local Skip", "block": "none",
        "sources": ["experiments/osram_meld_testbest_ua_20261014/BASELINE.md"],
        "updates_local": False,
        "initnotes": "Original zero-initialized Flat final projection; no Block2 adapter.",
    },
    "B1": {
        "name": "Graph Matching Base/Gap pairs", "block": "graph_matching_base_gap_pairs",
        "sources": ["docs/osram_new40_structure_cards.json",
                    "https://proceedings.mlr.press/v97/li19d.html"],
        "updates_local": False,
        "initnotes": "Zero shared value decoder; Local conditions paired graphs but is returned unchanged.",
    },
    "B2": {
        "name": "Dual sensory/symbolic relational attention", "block": "dual_attention_symbolic_relations",
        "sources": ["experiments/osram_meaningful20_round2_20261004/attention_candidates.json",
                    "https://proceedings.mlr.press/v267/altabaa25a.html"],
        "updates_local": True,
        "initnotes": "Zero Local and per-head memory outputs; the sensory/symbolic core starts nonzero.",
    },
    "B3": {
        "name": "MBT synchronous role bottleneck", "block": "mbt_role_bottleneck",
        "sources": ["experiments/osram_meaningful20_round2_20261004/representation_candidates.json",
                    "https://arxiv.org/abs/2107.00135"],
        "updates_local": True,
        "initnotes": "Zero Local/per-head decoders around five role streams and two synchronous exchanges.",
    },
    "B4": {
        "name": "Original neural production systems", "block": "conditional_new_07_neural_production",
        "sources": ["docs/osram_new40_conditional_cards.json",
                    "https://arxiv.org/abs/2103.01937"],
        "updates_local": False,
        "initnotes": "Zero memory bridges; original TokenReadout has no local_bridge. Three hard/ST rule steps.",
    },
    "B5": {
        "name": "TokenLearner V1.1 + TokenFuser", "block": "tokenlearner_fuser_v11",
        "sources": ["experiments/osram_meaningful20_round2_20261004/representation_candidates.json",
                    "https://arxiv.org/abs/2106.11297"],
        "updates_local": True,
        "initnotes": "Zero raw decoders; Xavier token-axis mixer avoids two serial zero maps and dead learning.",
    },
    "B6": {
        "name": "Finite learned robust PCA evidence decomposition",
        "block": "learned_robust_pca_evidence_decomposition",
        "sources": ["experiments/osram_meaningful20_round2_20261004/structured_candidates.json",
                    "https://arxiv.org/abs/2110.05649"],
        "updates_local": False,
        "initnotes": "Zero gamma makes Y-tanh(gamma)*sparse identity; detached SVD, fourteen learned updates.",
    },
}

HPO_KEYS = frozenset(("learning_rate", "weight_decay", "dropout"))

_FIXED = {
    "dataset": "MELD", "seed": 66, "batch_size": 32, "backbone_type": "osram",
    "training_objective": "emotion-only", "missing_mask_protocol": "gcnet-random-mask-v1",
    "train_rate_mode": "cyclic", "checkpoint_selection": "test-oracle-per-rate",
    "osram_num_heads": 8, "osram_key_dim": 64, "osram_value_dim": 64,
    "osram_write_step": 0.6, "osram_output_dim": 1600, "osram_adapter_hidden_dim": 0,
    "osram_bidirectional": False, "osram_forward_slot_reuse": False,
    "osram_readout_fusion": "flat", "osram_meaningful_block": "none",
}
_FORBIDDEN = (
    "osram_disable_local_skip", "osram_memory_only_adapter", "osram_local_skip_gate",
    "osram_history_input_gate", "osram_local_evidence_gate", "osram_relation_block",
    "osram_relation_dual_readout", "osram_history_query_adapter", "osram_post_grn",
    "osram_gap_increment_filter", "osram_decision_correction", "osram_hierarchical_evidence_gate",
    "osram_hierarchical_feature_only", "osram_history_recency", "osram_causal_gru_history",
    "osram_base_first_gap", "osram_ced_block", "paired_history_views", "completion_write_to_memory",
    "classification_completion", "local_context_residual", "node_interaction_residual", "text_core",
)


def build_config(base_dict, method, output, overrides=None, epochs=50):
    """Copy the historical TrainConfig; ``output`` is managed by the caller.

    No output/provenance or feature-root keys are added to TrainConfig. Historical
    inert fields (including jepa_weight) are preserved: emotion-only disables JEPA.
    Every run, including each Optuna trial, retains the 50-epoch budget.
    """
    if method not in METHODS:
        raise ValueError("Unknown finite Block2 method: " + str(method))
    if type(epochs) is not int or epochs != 50:
        raise ValueError("Block2 epochs must be 50 for both screen and HPO")
    for key, expected in _FIXED.items():
        if base_dict.get(key) != expected:
            raise ValueError("Baseline protocol mismatch: " + key)
    if list(base_dict.get("train_missing_rates", ())) != [i / 10 for i in range(8)]:
        raise ValueError("Baseline requires all eight cyclic train_missing_rates")
    for key in _FORBIDDEN:
        if base_dict.get(key, False):
            raise ValueError("Incompatible baseline intervention: " + key)
    for key in ("osram_readout_candidate", "osram_counterfactual_readout", "core20_method", "completion_path"):
        if base_dict.get(key, "none") != "none":
            raise ValueError("Incompatible baseline intervention: " + key)
    for key in ("initial_backbone_checkpoint", "b2_base_checkpoint", "b2_pretrain_checkpoint",
                "joint_pretrain_checkpoint", "teacher_checkpoint", "text_subspace_checkpoint"):
        if base_dict.get(key) is not None:
            raise ValueError("From-scratch baseline required: " + key)
    overrides = {} if overrides is None else dict(overrides)
    unexpected = set(overrides) - HPO_KEYS
    if unexpected:
        raise ValueError("Only learning_rate/weight_decay/dropout HPO is authorized: " + str(sorted(unexpected)))
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("Nonfinite or nonnumeric HPO value: " + key)
        if ((key == "learning_rate" and value <= 0) or (key == "weight_decay" and value < 0)
                or (key == "dropout" and not 0 <= value < 1)):
            raise ValueError("Invalid HPO value: " + key)
    config = deepcopy(base_dict)
    config.update(overrides)
    config.update(epochs=epochs, osram_meaningful_block=METHODS[method]["block"],
                  checkpoint_selection_metric="accuracy", device="cuda:0",
                  preserve_selection_artifacts=True)
    return config
