# Public tree manifest

Release candidate 0.1.0-rc1, derived from the canonical freeze package
FREEZE_QWEN_OVERNIGHT_20260907. Hash verification: `./scripts/verify_freeze.sh` (public-tree
SHA256SUMS) and `sha256sum -c freeze/package_SHA256SUMS` (against the private package, after
re-assembling its layout).

| public path | canonical counterpart | class |
|---|---|---|
| README.md, CHANGELOG.md, CITATION.cff, LICENSE | derived from package docs | publication text |
| docs/*.md | derived from package docs and receipts | publication text |
| figures/*.svg | derived from receipt tables only | generated figures |
| figures/kolmo_field_density.ppm | stoch-engine-0907:stoch/kolmo_field_density.ppm | retained artifact |
| figures/kolmo_field_density.png | lossless conversion | retained artifact |
| stoch/ | lane_a_stoch.patch (sources) | engine sources |
| src/cuda/qf_qsa_index.cu, src/cuda/qf.cu | lane_b_b1_b2.patch (target files) | patch-derived sources |
| src/main.cu | lane_a_stoch.patch (target file) | patch-derived sources |
| patches/lane_a/lane_a_stoch.patch | package lane_a_stoch.patch | patch |
| patches/lane_b/lane_b_b1_b2.patch | package lane_b_b1_b2.patch | patch |
| patches/amd_pending/ | package tests/ | pending AMD work |
| receipts/*.md | package receipts/ (sanitized) | receipts |
| scripts/*.sh | new; reproduce the public gates | tooling |
| freeze/package_SHA256SUMS | package SHA256SUMS | provenance evidence |

Numbers cited anywhere in this tree must map to a receipt; the audit table is reproduced in
docs/REPRODUCIBILITY.md (claim-to-receipt mapping).
