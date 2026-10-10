"""User-specified comparisons. Historical mechanisms are not counted as novel."""
PRIORITY_FAMILIES = {
    'relations': ('m01_sab', 'm02_dat', 'm03_gatv2', 'm04_edgeconv', 'm05_pna',
                  'm36_hopfield', 'm37_entmax_sab', 'm38_soft_moe', 'm39_node', 'm40_capsule'),
    'bilinear': ('m06_tfn', 'm07_lmf', 'm08_mfb', 'm09_tucker', 'm10_block'),
    'conditioning': ('m11_film', 'm12_cross_stitch', 'm13_mmtm', 'm14_gct', 'm15_dynamic_relu'),
    'pooling_geometry': ('m16_deep_sets', 'm17_janossy', 'm18_fspool', 'm19_netvlad',
                         'm20_attentive_statistics', 'm26_dolg', 'm27_isqrt_cov',
                         'm28_xcit_xca', 'm29_set_norm'),
    'crosses': ('m21_nfm', 'm22_dcnv2', 'm23_cin', 'm24_afm', 'm25_kan'),
    'mixers': ('m31_mlp_mixer', 'm32_gmlp_sgu', 'm33_fnet', 'm34_convnext_grn', 'm35_dynamixer'),
}
PRIORITY_INPUT_METHODS = ('m11_film', 'm13_mmtm')
# Placement controls do not inflate the historical forty-method catalog.
PRIORITY_INPUT_VARIANTS = ('m28_xcit_xca_direct',)
PRIORITY_NORM_METHODS = ('m30_dyt',)
PRIORITY_RESIDUAL_METHODS = tuple(sorted(name for names in PRIORITY_FAMILIES.values()
    for name in names if name not in PRIORITY_INPUT_METHODS))
PRIORITY_METHODS = tuple(sorted(PRIORITY_RESIDUAL_METHODS + PRIORITY_INPUT_METHODS + PRIORITY_NORM_METHODS))
