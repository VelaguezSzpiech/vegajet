# VegaJet `ops/`

Device-independent execution descriptors for gfx906 kernels.

- `op_desc.hpp` defines `OpKind`, `OpDesc` and `OpGraph`. Model plugins emit a flat `OpGraph` per step.
- `kernel_symbol(OpKind)` is the stable code-object symbol name the GPU backend resolves (`vj_gemv_q8_dp4a`,
  `vj_gdn_chunk_wy`, ...). A code object is accepted only if it exports every symbol the graph uses and its
  `.amdhsa.kernels` metadata reports `private_segment_fixed_size == 0` (checked by `core/grid_planner`).

Kernel sources themselves are not part of the CPU build.
