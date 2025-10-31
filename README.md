# Signer-Conditioned Attention ST-GCN with multi-head bias-aware gating

A PyTorch implementation for sign language detection and recognition using Spatio-Temporal Graph Convolutional Networks (ST-GCN). This model incorporates signer-aware attention, adversarial training to reduce signer bias, and supports both multi-class sign classification and binary signing detection.

## Features

- **ST-GCN Backbone**: Spatio-temporal graph convolutional layers for modeling sign language gestures
- **Bias-Aware Attention**: Attention mechanism conditioned on signer embeddings to adapt to different signing styles
- **Adversarial Signer Head**: Optional domain-adversarial training via gradient reversal layer to reduce signer-specific bias
- **Supervised Contrastive Learning**: Optional InfoNCE-based contrastive loss for better feature learning
- **Flexible Landmark Support**: Works with pose, hands, and face landmarks (e.g., from MediaPipe or OpenPose)
- **Pseudo-Signer Clustering**: K-means clustering of pose statistics to create pseudo signer IDs for unsupervised signer diversity
- **Binary & Multi-Class**: Supports both binary classification (signing vs. not-signing) and multi-class sign recognition

## Architecture

The model consists of several key components:

1. **ST-GCN Backbone**: Processes spatio-temporal keypoint sequences using graph convolutions
2. **Signer Encoder**: Encodes per-window pose statistics into signer embeddings
3. **Bias-Aware Attention**: Temporally attends to features while being conditioned on signer style
4. **Temporal Aggregator**: BiGRU-based aggregation over time
5. **Classifier Head**: Final classification layer
6. **Signer Head** (optional): Adversarial head for signer identification to encourage signer-invariant features

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- PyTorch >= 2.1.0
- NumPy >= 1.23
- scikit-learn >= 1.3
- matplotlib >= 3.7
- tqdm >= 4.65

Optional (for landmark extraction):
- MediaPipe >= 0.10.0
- OpenCV >= 4.7.0

## Data Format

The model expects:

1. **Landmark files**: `.npz` files containing keypoint sequences. Each file should contain arrays for:
   - `pose`: (T, 33, C) - Body pose landmarks
   - `left_hand`: (T, 21, C) - Left hand landmarks (optional)
   - `right_hand`: (T, 21, C) - Right hand landmarks (optional)
   - `face`: (T, 478, C) - Face landmarks (optional)

2. **Groundtruth file**: Text file with format:
   ```
   <video_id> <frame_idx> <label>
   ```

Example: s means signing and n means no signing
```
video001 0 s
video001 25 s
video001 50 n
```

## Usage

### Training

#### Basic Training

```bash
python -m train_landmarks \
    --data /path/to/landmarks/folder \
    --batch 32 \
    --epochs 50 \
    --lr 1e-3 \
    --num-classes 100 \
    --window 25 \
    --stride 1 \
    --include-pose \
    --include-hands \
    --save-best \
    --log-csv \
    --out runs/signgcn
```

#### Binary Classification (Signing Detection)

```bash
python -m train_landmarks \
    --data /path/to/landmarks/folder \
    --binary \
    --signing-labels "sign,signing,gesture" \
    --batch 32 \
    --epochs 50 \
    --best-metric pr_auc \
    --out runs/signgcn_binary
```

#### With Pseudo-Signer Clustering

Uses K-means clustering on pose statistics to create pseudo signer IDs for adversarial training:

```bash
python -m train_landmarks \
    --data /path/to/landmarks/folder \
    --use-pseudo-signers \
    --num-pseudo-signers 8 \
    --signer-loss-weight 0.5 \
    --batch 32 \
    --epochs 50 \
    --out runs/signgcn_pseudo
```

#### With Supervised Contrastive Learning

Adds InfoNCE-based contrastive loss:

```bash
python -m train_landmarks \
    --data /path/to/landmarks/folder \
    --use-supcon \
    --supcon-weight 0.1 \
    --supcon-temp 0.07 \
    --batch 32 \
    --epochs 50 \
    --out runs/signgcn_supcon
```

#### Using Pre-defined Splits

```bash
# First, create splits
python -m build_video_splits \
    --data /path/to/landmarks/folder \
    --window 25 \
    --stride 1 \
    --train 0.8 \
    --val 0.1 \
    --test 0.1 \
    --out /path/to/landmarks/folder/splits.json

# Then train with splits
python -m train_landmarks \
    --data /path/to/landmarks/folder \
    --splits-json /path/to/landmarks/folder/splits.json \
    --batch 32 \
    --epochs 50 \
    --out runs/signgcn_splits
```

### Key Training Arguments

- `--data`: Path to folder containing `.npz` files and groundtruth
- `--window`: Temporal window size (default: 25)
- `--stride`: Sliding window stride (default: 1)
- `--coords`: Number of coordinates per keypoint (default: 2, i.e., x, y)
- `--include-pose`: Include body pose landmarks
- `--include-hands`: Include hand landmarks
- `--include-face`: Include face landmarks
- `--batch`: Batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--num-classes`: Number of sign classes (for multi-class)
- `--binary`: Train binary classifier (signing vs. not-signing)
- `--use-pseudo-signers`: Enable pseudo-signer clustering
- `--use-supcon`: Enable supervised contrastive learning
- `--save-best`: Save best model checkpoint
- `--best-metric`: Metric for best model selection (`acc`, `loss`, `pr_auc`, `f1`)
- `--log-csv`: Log metrics to CSV file
- `--out`: Output directory for checkpoints and logs

### Visualization

Visualize signer embeddings with t-SNE or PCA:

```bash
python -m visualize_signer_embeddings \
    --data /path/to/landmarks/folder \
    --ckpt runs/signgcn/best.pt \
    --window 25 \
    --include-pose \
    --include-hands \
    --reduce tsne \
    --out embeddings_plot.png \
    --save-csv embeddings.csv
```

### Building Video Splits

Create train/val/test splits based on videos:

```bash
python -m build_video_splits \
    --data /path/to/landmarks/folder \
    --window 25 \
    --stride 1 \
    --s-label S \
    --rule any \
    --train 0.8 \
    --val 0.1 \
    --test 0.1 \
    --out splits.json
```

## Model Configuration

The model can be customized through various hyperparameters:

- `stgcn_channels`: Hidden dimensions for ST-GCN layers (default: `(64, 128, 128)`)
- `stgcn_kernel`: Temporal kernel size (default: `3`)
- `stgcn_dilations`: Dilation rates for temporal convolutions (default: `(1, 2, 3)`)
- `temporal_hidden`: Hidden dimension for temporal aggregator (default: `256`)
- `signer_emb_dim`: Dimension of signer embedding (default: `64`)
- `attn_heads`: Number of attention heads (default: `4`)
- `lambda_grl`: Gradient reversal weight for adversarial training (default: `0.5`)
- `dropout`: Dropout rate (default: `0.1`)

## Output

Training produces:

- `best.pt`: Best model checkpoint (if `--save-best` is used)
- `last.pt`: Last epoch checkpoint
- `metrics.csv`: Training metrics per epoch (if `--log-csv` is used)

Checkpoint format:
```python
{
    'model': model.state_dict(),
    'epoch': epoch_number,
    'val_acc': validation_accuracy,
    # ... other metadata
}
```

## Project Structure

```
SignSTGCN/
├── train_landmarks.py          # Main training script
├── signGCN.py                  # Model entry point / example usage
├── model.py                    # SignSTGCNModel implementation
├── losses.py                   # Loss functions
├── build_video_splits.py       # Video split generation
├── visualize_signer_embeddings.py  # Embedding visualization
├── datasets/
│   └── landmarks_npz.py        # NPZ landmark dataset
├── layers/
│   └── graph.py                # ST-GCN backbone
├── modules/
│   ├── signer.py               # Signer encoder
│   ├── attention.py            # Bias-aware attention
│   ├── temporal.py             # Temporal aggregator
│   └── heads.py                # Classification and signer heads
└── utils/
    └── graph.py                # Graph utilities
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{signstgcn,
  title={SignSTGCN: Sign Language Detection with Spatio-Temporal Graph Convolutional Networks},
  author={Cyrine Chaabani},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation builds upon ST-GCN architectures and incorporates techniques for reducing signer bias in sign language recognition models.
