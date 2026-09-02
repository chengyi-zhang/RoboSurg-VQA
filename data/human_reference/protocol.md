# Human annotation protocol

The audit covered 250 frames. A representative stratum sampled eight
temporally distributed frames from each video key (200 frames). A separate
challenge stratum contained 50 frames selected for rare candidate labels, low
candidate confidence, or disagreement with the deterministic mask rule.

Two authors annotated every frame independently in different random orders.
Candidate labels and source identities were hidden. A third author adjudicated
disagreements without changing agreed labels. The deterministic central-mask
target was computed separately from the official masks.

## Fields

- `q4_bleeding`: visible bleeding (`yes`, `no`, `uncertain`)
- `q4_smoke`: visible smoke (`yes`, `no`, `uncertain`)
- `q4_occlusion`: meaningful obstruction of the surgical view (`yes`, `no`, `uncertain`)
- `q5_value`: overall image quality (`clear`, `blurry`, `reflective`, `mixed`, `uncertain`)
- `q6_glare`: focal saturated or mirror-like glare (`yes`, `no`, `uncertain`)
- `q7_contrast_normal`: whether tissue and instrument boundaries remain distinguishable (`yes`, `no`, `uncertain`)
- `q9_smoke_region`: smoke location, `none`, `smoke present but unlocalizable`, or `uncertain`
- `global_unusable_frame_yes_no`: frame cannot be judged reliably

For Q5 evaluation, `blurry`, `reflective`, and `mixed` are harmonised as
`degraded`. The original annotations remain available in the CSV files.
