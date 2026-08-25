"""Zero-parameter relation-track routing for the locked GCNet graph core.

This is a custom GCNet hypothesis.  It is deliberately not presented as an
adaptation of MrMP: relation contributions remain separate between the two
existing graph layers, then only matching relation tracks are propagated.
"""

import torch


def _validate_rgcn(conv):
    if conv.num_bases is not None or conv.num_blocks is not None:
        raise ValueError("relation tracks require an undecomposed RGCNConv")
    if conv.aggr != "mean":
        raise ValueError("relation tracks require RGCNConv aggr='mean'")
    if not isinstance(conv.in_channels, int):
        raise ValueError("relation tracks require homogeneous RGCNConv inputs")


def _graph_conv_linears(conv):
    neighbor = getattr(conv, "lin_l", None)
    if neighbor is None:
        neighbor = conv.lin_rel
    root = getattr(conv, "lin_r", None)
    if root is None:
        root = conv.lin_root
    return neighbor, root


def _propagate_graph_conv(conv, values, edge_index, node_count):
    return conv.propagate(
        edge_index,
        x=(values, values),
        edge_weight=None,
        size=(node_count, node_count),
    )


def decompose_rgcn_relation_tracks(conv, x, edge_index, edge_type):
    """Return common, per-relation messages, and their native RGCN sum."""
    _validate_rgcn(conv)
    if not isinstance(edge_index, torch.Tensor):
        raise ValueError("relation tracks require dense edge_index")
    if not torch.is_floating_point(x):
        raise ValueError("relation tracks require floating-point node features")

    node_count = x.size(0)
    size = (node_count, node_count)
    tracks = []
    for relation in range(conv.num_relations):
        relation_edges = edge_index[:, edge_type == relation]
        mean = conv.propagate(
            relation_edges,
            x=x,
            edge_type_ptr=None,
            size=size,
        )
        tracks.append(mean @ conv.weight[relation])
    tracks = torch.stack(tracks, dim=0)

    common = x.new_zeros((node_count, conv.out_channels))
    if conv.root is not None:
        common = common + x @ conv.root
    if conv.bias is not None:
        common = common + conv.bias
    fused = common + tracks.sum(dim=0)
    return common, tracks, fused


def relation_track_graph_conv(
    conv,
    common,
    tracks,
    fused,
    edge_index,
    edge_type,
    diagonal=True,
):
    """Apply full or diagonal relation transitions through an existing GraphConv."""
    if conv.aggr != "add":
        raise ValueError("relation-track routing requires GraphConv aggr='add'")
    if not isinstance(edge_index, torch.Tensor):
        raise ValueError("relation tracks require dense edge_index")
    if edge_type.numel() and tracks.size(0) <= int(edge_type.max().item()):
        raise ValueError("edge_type exceeds the number of relation tracks")

    node_count = fused.size(0)
    neighbor = _propagate_graph_conv(conv, common, edge_index, node_count)
    if diagonal:
        for relation in range(tracks.size(0)):
            relation_edges = edge_index[:, edge_type == relation]
            neighbor = neighbor + _propagate_graph_conv(
                conv, tracks[relation], relation_edges, node_count
            )
    else:
        for input_relation in range(tracks.size(0)):
            for output_relation in range(tracks.size(0)):
                relation_edges = edge_index[:, edge_type == output_relation]
                neighbor = neighbor + _propagate_graph_conv(
                    conv,
                    tracks[input_relation],
                    relation_edges,
                    node_count,
                )

    lin_neighbor, lin_root = _graph_conv_linears(conv)
    return lin_neighbor(neighbor) + lin_root(fused)
