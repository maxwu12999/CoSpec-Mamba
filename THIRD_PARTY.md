# Third-party sources

The MIT license at the repository root covers original CoSpec code. It does not
relicense external model implementations. External files are not committed;
`tools/fetch_upstreams.py` downloads pinned source files and verifies SHA-256
before placing them in the ignored `frozen/**/vendor` directories.

| Component | Original source | Pinned commit |
|---|---|---|
| ConvMamba | https://github.com/zhysess/ConvMamba | `aa71fe5ab9abe4606383df0c773941a142ee59fe` |
| HybridSN | https://github.com/Pancakerr/HybridSN | `01b3ba919d1b3d8d15b132ff3e804ea0488ca08f` |
| A²S²K | https://github.com/suvojit-0x55aa/A2S2K-ResNet | `0bba673d6343bd147b0d29a93d3a4d7a9bc522d9` |
| SQSFormer | https://github.com/chenning0115/SQSFormer | `4baf5d33254ca5fef56edbb25c0d2b04ae987f4d` |
| GSCViT | https://github.com/flyzzie/TGRS-GSC-VIT | `2ac8719321e1ef67238e9e8b77f934e604172085` |
| MiM | https://github.com/zhouweilian1904/Mamba-in-Mamba | `ace0c0e1b1b7af7ade409a641a2134cdf699ce97` |
| VisionMamba dependency | https://github.com/kyegomez/VisionMamba | `c88e09a52240910e08c4af754d340ad946112487` |
| zeta dependency | https://github.com/kyegomez/zeta | `fe82c50e0f738ff6b4ae37ea4f3af6631b880c74` |
| parallel scan dependency | https://github.com/alxndrTL/mamba.py | `5c4136c2c28bc10d5d6655aa81b7a60051b50f00` |

Source licenses are fetched alongside files where an upstream license file is
recorded. A public GitHub repository does not by itself grant an MIT license.
Consult each original project for reuse conditions. The complete file-level
mapping is in `tools/upstream_sources.json` and the archived
`frozen/external/UPSTREAM.json` manifests.

The HybridSN class is extracted verbatim from the pinned notebook; its function
bodies are not edited. The adapter restores the same source bytes used in the
experiment archive. External loaders select definitions with Python AST to
avoid running unrelated upstream training entry points; see the archived
adapter implementation for shape and fused-scan compatibility handling.

CUDA execution also depends on the separately installed projects
[Mamba](https://github.com/state-spaces/mamba) and
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d). These packages retain
their own licenses and build requirements.

