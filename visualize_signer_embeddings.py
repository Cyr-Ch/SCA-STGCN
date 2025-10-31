from __future__ import annotations
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import SignSTGCNModel
from .datasets import NpzLandmarksDataset
from .utils import build_hand_body_adjacency


def collate_pad(batch):
    Xs, stats, ys, metas = zip(*batch)
    return torch.stack(Xs, 0), torch.stack(stats, 0), torch.stack(ys, 0), list(metas)


def main():
    ap = argparse.ArgumentParser(description="Visualize signer embeddings with t-SNE/PCA")
    ap.add_argument("--data", required=True, help="Path to landmarks folder containing .npz and groundtruth")
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint (best.pt/last.pt)")
    ap.add_argument("--id-list", default=None, help="Optional path to txt with video IDs to include (one per line)")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--coords", type=int, default=2)
    ap.add_argument("--include-pose", action="store_true")
    ap.add_argument("--no-include-pose", dest="include_pose", action="store_false")
    ap.set_defaults(include_pose=True)
    ap.add_argument("--include-hands", action="store_true")
    ap.add_argument("--include-face", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reduce", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--out", default=None, help="Optional path to save PNG plot (otherwise will show)")
    ap.add_argument("--save-csv", default=None, help="Optional path to save embeddings CSV")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_path_txt = os.path.join(args.data, "groundtruth.txt")
    gt_path = gt_path_txt if os.path.exists(gt_path_txt) else os.path.join(args.data, "groundtruth")

    allowed_ids = None
    if args.id_list and os.path.exists(args.id_list):
        with open(args.id_list, "r", encoding="utf-8") as f:
            allowed_ids = set([line.strip() for line in f if line.strip()])

    ds = NpzLandmarksDataset(
        root=args.data,
        gt_path=gt_path,
        window=args.window,
        stride=args.stride,
        in_coords=args.coords,
        include_pose=bool(args.include_pose),
        include_hands=bool(args.include_hands),
        include_face=bool(args.include_face),
        allowed_ids=allowed_ids,
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty with the given filters.")

    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_pad)

    sample_X, sample_stats, _, _ = ds[0]
    T, J, C = sample_X.shape
    hand_edges = tuple((i, i + 1) for i in range(0, max(0, J - 1)))
    A = build_hand_body_adjacency(J, hand_edges, tuple()).to(device)

    model = SignSTGCNModel(
        num_joints=J,
        in_coords=C,
        num_classes=2,
        signer_stats_dim=sample_stats.numel(),
        use_signer_head=False,
    ).to(device).eval()

    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    signer_embs = []
    labels = []
    videos = []
    with torch.no_grad():
        for X, stats, y, meta in dl:
            X = X.to(device)
            stats = stats.to(device)
            out = model(X, A, stats, return_features=True)
            signer_embs.append(out["signer_emb"].detach().cpu().numpy())
            labels += [m.get("label", "") for m in meta]
            videos += [m.get("video", "") for m in meta]

    signer_embs = np.concatenate(signer_embs, axis=0)

    # Optional CSV dump
    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "label"] + [f"e{i}" for i in range(signer_embs.shape[1])])
            for v, l, row in zip(videos, labels, signer_embs):
                w.writerow([v, l] + list(map(float, row)))

    # Dimensionality reduction
    if args.reduce == "tsne":
        try:
            from sklearn.manifold import TSNE
        except Exception:
            raise RuntimeError("scikit-learn is required for t-SNE. Install with: pip install scikit-learn")
        z2 = TSNE(n_components=2, perplexity=float(args.perplexity), random_state=0).fit_transform(signer_embs)
    else:
        try:
            from sklearn.decomposition import PCA
        except Exception:
            raise RuntimeError("scikit-learn is required for PCA. Install with: pip install scikit-learn")
        z2 = PCA(n_components=2, random_state=0).fit_transform(signer_embs)

    # Plot
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise RuntimeError("matplotlib is required to plot. Install with: pip install matplotlib")

    colors = ["tab:blue" if str(l).lower() == "s" else "tab:gray" for l in labels]
    plt.figure(figsize=(7, 6))
    plt.scatter(z2[:, 0], z2[:, 1], s=10, c=colors, alpha=0.8)
    plt.title("Signer embeddings ({})".format(args.reduce))
    plt.axis("off")
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()


