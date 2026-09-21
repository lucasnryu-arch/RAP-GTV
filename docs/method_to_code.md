# Method-to-code map

| Paper component | Public implementation |
|---|---|
| Deformation representation | `rapgtv.representation.deformation.build_deformation_representation` |
| Metadata quality mapping (`q_meta`) | `rapgtv.reliability.node.compute_metadata_reliability` |
| Residual stability (`q_res`) | `rapgtv.reliability.node.compute_residual_reliability` |
| Mean-one fidelity weights (`rho`) | `rapgtv.reliability.node.compute_node_reliability` |
| Terrain graph | `rapgtv.terrain.graph.build_physical_graph` |
| Optical representation (`z_O`) | `rapgtv.optical.core.build_optical_representation` |
| Optical scale (`sigma_O`) and gate | `rapgtv.optical.core.compute_optical_gate` |
| Edge conductance normalization | `rapgtv.graph.weights.compute_edge_conductance` |
| Characteristic deformation jump | `rapgtv.graph.weights.compute_jump_scale` |
| Graph-TV objective and solver | `rapgtv.solvers.recovery.compute_rap_gtv_objective`, `solve_rap_gtv` |
| Deformation zoning | `rapgtv.clustering.kmeans.kmeans_fit` |
| `K*` selection | `rapgtv.clustering.selection.select_k` |
| Validation selection | `rapgtv.selection.real_selector.select_hyperparameters` |
| FDD | `rapgtv.metrics.real.compute_fdd` |
| FI and coordinate-only graph | `rapgtv.metrics.spatial.build_evaluation_graph`, `compute_fi` |
| `k_s`/`lambda` grid | `configs/real_common.json` |
| Site-selected parameters | `configs/xiongba.json`, `offida.json`, `mendatica.json`, `zhouqu_xieliupo.json` |

The primary FI graph uses coordinate-only symmetric nearest neighbours with `k_eval = 8`. Site configuration is selected from validation metrics; test observations are used only for final FDD evaluation.
