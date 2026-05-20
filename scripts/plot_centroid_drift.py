import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333',
    'text.color': '#222',
    'axes.labelcolor': '#222',
    'xtick.color': '#333',
    'ytick.color': '#333',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.color': '#ddd',
    'grid.alpha': 0.7,
})

BAR_COLORS = ['#E74C3C', '#1ABC9C', '#2E86C1', '#27AE60', '#F39C12',
              '#8E44AD', '#16A085', '#D4AC0D']


def load_prototypes(exp_dir, task_id):
    path = os.path.join(exp_dir, 'features', f'task_{task_id}', 'prototypes.pt')
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu')


def compute_drift(proto_ref, proto_later):
    n_classes = min(proto_ref.shape[0], proto_later.shape[0])
    D_shared = min(proto_ref.shape[1], proto_later.shape[1])
    drift = torch.norm(
        proto_later[:n_classes, :D_shared] - proto_ref[:n_classes, :D_shared],
        p=2, dim=1
    )
    return drift.numpy()


def plot_drift_comparison(exp_dirs, exp_labels, ref_task, target_tasks,
                          output_path):
    n_targets = len(target_tasks)
    fig, axes = plt.subplots(1, n_targets,
                             figsize=(max(8, 5 * n_targets), 5),
                             squeeze=False)

    for t_idx, target_task in enumerate(target_tasks):
        ax = axes[0][t_idx]

        all_drifts = []
        valid_labels = []
        n_classes = None

        for exp_dir, exp_label in zip(exp_dirs, exp_labels):
            proto_ref = load_prototypes(exp_dir, ref_task)
            proto_target = load_prototypes(exp_dir, target_task)

            if proto_ref is None or proto_target is None:
                continue

            drift = compute_drift(proto_ref, proto_target)
            all_drifts.append(drift)
            valid_labels.append(exp_label)
            if n_classes is None:
                n_classes = len(drift)
            else:
                n_classes = min(n_classes, len(drift))

        if len(all_drifts) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    color='#888', fontsize=14, transform=ax.transAxes)
            continue

        # Trim to common class count
        all_drifts = [d[:n_classes] for d in all_drifts]
        n_exps = len(all_drifts)

        # Grouped bar chart
        x = np.arange(n_classes)
        bar_width = 0.8 / n_exps
        offsets = np.linspace(-(n_exps - 1) / 2 * bar_width,
                              (n_exps - 1) / 2 * bar_width, n_exps)

        for i, (drift, label) in enumerate(zip(all_drifts, valid_labels)):
            color = BAR_COLORS[i % len(BAR_COLORS)]
            bars = ax.bar(x + offsets[i], drift, bar_width * 0.9,
                          label=label, color=color, alpha=0.85,
                          edgecolor='none')

        ax.set_xlabel('Class Index', fontsize=12)
        ax.set_ylabel('Centroid Drift (L2)', fontsize=12)
        ax.set_title(f'Drift: Task {ref_task} → Task {target_task}',
                     fontsize=13, fontweight='bold', color='#222')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(n_classes)],
                           fontsize=7, rotation=45 if n_classes > 15 else 0)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.4,
                  labelcolor='#333')

        # Annotate average drift per experiment
        for i, (drift, label) in enumerate(zip(all_drifts, valid_labels)):
            avg = drift.mean()
            color = BAR_COLORS[i % len(BAR_COLORS)]
            ax.axhline(y=avg, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax.text(n_classes - 0.5, avg, f'avg={avg:.4f}',
                    color=color, fontsize=8, va='bottom', ha='right')

    plt.tight_layout(pad=2.0)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved centroid drift plot to: {output_path}")
    plt.close(fig)


def plot_drift_over_tasks(exp_dirs, exp_labels, ref_task, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (exp_dir, exp_label) in enumerate(zip(exp_dirs, exp_labels)):
        proto_ref = load_prototypes(exp_dir, ref_task)
        if proto_ref is None:
            continue

        # Find all available tasks
        feat_dir = os.path.join(exp_dir, 'features')
        task_ids = sorted([
            int(name.split('_')[1])
            for name in os.listdir(feat_dir)
            if name.startswith('task_')
        ])

        tasks_plot, avg_drifts = [], []
        for t in task_ids:
            if t <= ref_task:
                continue
            proto_t = load_prototypes(exp_dir, t)
            if proto_t is None:
                continue
            drift = compute_drift(proto_ref, proto_t)
            tasks_plot.append(t)
            avg_drifts.append(drift.mean())

        if len(tasks_plot) == 0:
            continue

        color = BAR_COLORS[i % len(BAR_COLORS)]
        ax.plot(tasks_plot, avg_drifts, marker='o', linewidth=2,
                markersize=8, label=exp_label, color=color, alpha=0.9)

        # Annotate last point
        ax.annotate(f'{avg_drifts[-1]:.4f}',
                    xy=(tasks_plot[-1], avg_drifts[-1]),
                    xytext=(5, 8), textcoords='offset points',
                    color=color, fontsize=9, fontweight='bold')

    ax.set_xlabel('Task Index', fontsize=13)
    ax.set_ylabel('Average Centroid Drift (L2)', fontsize=13)
    ax.set_title(f'Centroid Drift Over Tasks (ref = Task {ref_task})',
                 fontsize=14, fontweight='bold', color='#222')
    ax.legend(fontsize=10, framealpha=0.4, labelcolor='#333')

    # Save
    base, ext = os.path.splitext(output_path)
    line_path = f"{base}_over_tasks{ext}"
    os.makedirs(os.path.dirname(line_path) if os.path.dirname(line_path) else '.', exist_ok=True)
    fig.savefig(line_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved drift-over-tasks plot to: {line_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Centroid drift visualization for MEMO experiments')
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='Log directories for each experiment')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='Display names for each experiment')
    parser.add_argument('--ref_task', type=int, default=0,
                        help='Reference task index (prototypes computed after this task)')
    parser.add_argument('--target_tasks', nargs='+', type=int, default=None,
                        help='Target tasks to measure drift against (default: all after ref)')
    parser.add_argument('--output', type=str, default='figures/centroid_drift.png',
                        help='Output image path')
    args = parser.parse_args()

    assert len(args.dirs) == len(args.labels), \
        "Number of --dirs must match number of --labels"

    # Auto-detect target tasks
    if args.target_tasks is None:
        all_tasks = set()
        for d in args.dirs:
            feat_dir = os.path.join(d, 'features')
            print(f"  Searching: {os.path.abspath(feat_dir)}")
            if os.path.isdir(feat_dir):
                for name in os.listdir(feat_dir):
                    if name.startswith('task_'):
                        t = int(name.split('_')[1])
                        if t > args.ref_task:
                            all_tasks.add(t)
                if not all_tasks:
                    print(f"    -> 'features/' exists but contains no task_* folders after ref_task {args.ref_task}")
            else:
                print(f"    -> 'features/' folder NOT found")
        args.target_tasks = sorted(all_tasks)
        print(f"Auto-detected target tasks: {args.target_tasks}")

    if len(args.target_tasks) == 0:
        print("\n[ERROR] No feature snapshots found!")
        print("Expected directory structure inside each --dirs path:")
        print("  <dir>/features/task_0/prototypes.pt")
        print("  <dir>/features/task_1/prototypes.pt")
        print("  ...")
        print("\nMake sure you:")
        print("  1. Have run training AFTER adding _save_feature_snapshot() to memo.py")
        print("  2. Pass the correct experiment log directory, e.g.:")
        print("     --dirs logs/benchmark/cifar100/memo/<exp_name>")
        return

    # Bar chart: per-class drift
    plot_drift_comparison(args.dirs, args.labels, args.ref_task,
                          args.target_tasks, args.output)

    # Line chart: average drift over tasks
    plot_drift_over_tasks(args.dirs, args.labels, args.ref_task,
                          args.output)


if __name__ == '__main__':
    main()
