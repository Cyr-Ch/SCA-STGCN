## SignSTGCN Experiment Summary

This document captures the current SignSTGCN variants trained in the repository, along with their architecture highlights and parameter counts (measured from the latest checkpoints).

### Core Architecture
- **Backbone:** 3-stage ST-GCN (`STGCNBackbone`) producing temporal features of size 128.
- **Signer Encoding & Attention:** `SignerEncoder` produces a signer embedding fed into either Bias-Aware or FiLM-based attention to modulate features before aggregation.
- **Temporal Aggregation:** Bi-directional GRU (`TemporalAggregator`) compresses sequence features.
- **Heads:** `ClassifierHead` for gesture classes and, optionally, a `SignerHead` (GRL) for pseudo-signer adversarial training. Loss modules enable MI minimization and supervised contrastive learning as needed.
- **Typical Input:** 25 frames × 553 joints × 2 coordinates (pose + both hands + face landmarks).

### Model Variants (from `runs/limited_test/*`)
| Model | Params | Notes |
|-------|--------|-------|
| `basic` | **1,089,749** | Baseline SignSTGCN, 3-class classifier (S/P/n). |
| `binary` | 1,085,645 | Same backbone with 2-class head (signing vs other). |
| `film_attention` | 1,114,837 | Replaces Bias-Aware attention with FiLM modulation (extra scale/shift MLPs). |
| `film_pseudo` | 1,248,217 | FiLM attention + signer-adversarial head & pseudo-signer clustering (largest variant). |
| `supcon` | 1,089,749 | Baseline architecture with supervised-contrastive loss (no structural change). |
| `supcon_film` | 1,114,837 | FiLM attention + supervised-contrastive loss. |
| `pseudo_signers` | 1,223,129 | Adds signer head trained adversarially using pseudo-signer clusters. |
| `mi_minimization` | 1,089,749 (+~41k for CLUB estimator during training) | Baseline architecture plus MI minimization head (CLUB estimator not saved). |
| `signgcn_3vids` (legacy) | 1,220,053 | Early experiment combining signer head and modified classifier. |

Parameter counts were computed by loading each `best.pt` checkpoint under `runs/limited_test/*` (script: `python -c "..."` using `torch.load`). `film_*` models incur ~25k extra params from FiLM layers; signer-adversarial variants add ~133k from the GRL head; SupCon/MI techniques mostly alter the loss pipeline rather than the graph backbone.

### Training Notes
- **Checkpoint Strategy:** `train_base.py` now writes `epoch_###.pt` every epoch and keeps `best.pt` as the top-performing weights. GCS uploads mirror the same naming when `--gcs-output` is set.
- **Manifest-based Loading:** `NpzLandmarksDataset` accepts `segment_list_file` for train/val/test splits, enabling curated manifests (e.g., `train_segments_list.txt`) to restrict training to specific segment subsets.
- **Mixed Segment Shapes:** The dataset zero-pads samples whose joint dimension is smaller than the first-seen segment, which allows mixing older pose-only data with newer pose+hands+face segments (though homogeneous manifests are recommended).

### Suggested Usage
- Use `train_basic.py --train-segment-list ... --val-segment-list ...` to train the baseline on curated sets.
- For FiLM experiments, use `train_film.py`; for pseudo-signer variants, run `train_pseudo_signers.py`; each script shares the same checkpoint policy and dataset behavior.
- Monitor signer fairness when using pseudo-signer or MI models, as they explicitly target bias mitigation.

_Last updated: {{DATE}} — regenerate parameter counts after new trainings._ 

