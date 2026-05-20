import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333',
    'text.color': '#222',
    'axes.labelcolor': '#222',
    'xtick.color': '#333',
    'ytick.color': '#333',
    'font.family': 'sans-serif',
    'font.size': 10,
})

PALETTE = [
    '#E74C3C', '#1ABC9C', '#2E86C1', '#27AE60', '#F39C12',
    '#8E44AD', '#16A085', '#D4AC0D', '#7D3C98', '#229954',
    '#C0392B', '#2980B9', '#E67E22', '#148F77', '#6C3483',
    '#B7950B', '#1F618D', '#1E8449', '#CB4335', '#5B2C6F',
]


def load_snapshot(exp_dir, task_id):
    base = os.path.join(exp_dir, 'features', f'task_{task_id}')
    feat_path = os.path.join(base, 'features.pt')
    label_path = os.path.join(base, 'labels.pt')
    proto_path = os.path.join(base, 'prototypes.pt')

    if not os.path.exists(feat_path):
        return None, None, None

    features = torch.load(feat_path, map_location='cpu').numpy()
    labels = torch.load(label_path, map_location='cpu').numpy()

    prototypes = None
    if os.path.exists(proto_path):
        prototypes = torch.load(proto_path, map_location='cpu').numpy()

    return features, labels, prototypes


def plot_tsne_grid(exp_dirs, exp_labels, task_ids, output_path,
                   perplexity=30, max_iter=1000, max_samples=1000,
                   show_prototypes=True):
    n_rows = len(exp_dirs)
    n_cols = len(task_ids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for row, (exp_dir, exp_label) in enumerate(zip(exp_dirs, exp_labels)):
        for col, task_id in enumerate(task_ids):
            ax = axes[row][col]
            features, labels, prototypes = load_snapshot(exp_dir, task_id)

            if features is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        color='#999', fontsize=14, transform=ax.transAxes)
                ax.set_title(f'{exp_label}\nTask {task_id}', fontsize=12,
                             fontweight='bold', color='#222')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if len(features) > max_samples:
                idx = np.random.choice(len(features), max_samples, replace=False)
                features = features[idx]
                labels = labels[idx]

            has_proto = (show_prototypes and prototypes is not None
                         and prototypes.shape[0] > 0)
            if has_proto:
                D_shared = min(features.shape[1], prototypes.shape[1])
                feat_trunc = features[:, :D_shared]
                proto_trunc = prototypes[:, :D_shared]
                combined = np.vstack([feat_trunc, proto_trunc])
                n_feat = len(feat_trunc)
                n_proto = len(proto_trunc)
            else:
                combined = features
                n_feat = len(features)
                n_proto = 0

            tsne = TSNE(n_components=2,
                        perplexity=min(perplexity, len(combined) - 1),
                        max_iter=max_iter, random_state=42, init='pca',
                        learning_rate='auto')
            emb = tsne.fit_transform(combined)

            emb_feat = emb[:n_feat]
            emb_proto = emb[n_feat:] if n_proto > 0 else None

            unique_labels = np.unique(labels)
            for i, cls in enumerate(unique_labels):
                mask = labels == cls
                color = PALETTE[int(cls) % len(PALETTE)]
                ax.scatter(emb_feat[mask, 0], emb_feat[mask, 1],
                           c=color, s=15, alpha=0.6, edgecolors='none',
                           label=f'C{int(cls)}')

            if emb_proto is not None:
                for c in range(n_proto):
                    color = PALETTE[c % len(PALETTE)]
                    ax.scatter(emb_proto[c, 0], emb_proto[c, 1],
                               marker='*', s=220, c=color,
                               edgecolors='black', linewidths=0.8,
                               zorder=10)
                    ax.annotate(f'{c}', (emb_proto[c, 0], emb_proto[c, 1]),
                                fontsize=6, fontweight='bold', color='#222',
                                xytext=(4, 4), textcoords='offset points',
                                zorder=11)

            ax.set_title(f'After Task {task_id}',
                         fontsize=12, fontweight='bold', color='#222')
            ax.set_xticks([])
            ax.set_yticks([])

            if col == n_cols - 1 and len(unique_labels) <= 15:
                ax.legend(fontsize=6, loc='upper right',
                          framealpha=0.3, labelcolor='#333',
                          markerscale=1.5, ncol=2)

        axes[row][0].set_ylabel(exp_label, fontsize=13, fontweight='bold',
                                color='#222', labelpad=10)

    plt.tight_layout(pad=2.0)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved t-SNE plot to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization for MEMO experiments')
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='Log directories for each experiment (containing features/ subfolder)')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='Display names for each experiment')
    parser.add_argument('--tasks', nargs='+', type=int, default=None,
                        help='Task indices to plot (default: all available)')
    parser.add_argument('--output', type=str, default='figures/tsne_comparison.png',
                        help='Output image path')
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Max exemplars per subplot (subsampled if more)')
    args = parser.parse_args()

    assert len(args.dirs) == len(args.labels), \
        "Number of --dirs must match number of --labels"

    if args.tasks is None:
        all_tasks = set()
        for d in args.dirs:
            feat_dir = os.path.join(d, 'features')
            print(f"  Searching: {os.path.abspath(feat_dir)}")
            if os.path.isdir(feat_dir):
                for name in os.listdir(feat_dir):
                    if name.startswith('task_'):
                        all_tasks.add(int(name.split('_')[1]))
                if not all_tasks:
                    print(f"    -> 'features/' exists but contains no task_* folders")
            else:
                print(f"    -> 'features/' folder NOT found")
        args.tasks = sorted(all_tasks)
        print(f"Auto-detected tasks: {args.tasks}")

    if len(args.tasks) == 0:
        print("\n[ERROR] No feature snapshots found!")
        print("Expected directory structure inside each --dirs path:")
        print("  <dir>/features/task_0/features.pt")
        print("  <dir>/features/task_0/labels.pt")
        print("  <dir>/features/task_1/features.pt")
        print("  ...")
        print("\nMake sure you:")
        print("  1. Have run training AFTER adding _save_feature_snapshot() to memo.py")
        print("  2. Pass the correct experiment log directory, e.g.:")
        print("     --dirs logs/benchmark/cifar100/memo/<exp_name>")
        return

    plot_tsne_grid(args.dirs, args.labels, args.tasks, args.output,
                   perplexity=args.perplexity, max_samples=args.max_samples)


if __name__ == '__main__':
    main()
