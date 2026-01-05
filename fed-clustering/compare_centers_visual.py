#!/usr/bin/env python3
"""Compare and visualize two sets of cluster centers.

Usage:
  python3 compare_centers_visual.py --local local_kmeans_centers.txt --remote nv_centers.txt \
    [--local-feats local_feats.json] [--remote-feats remote_feats.json] [--out-dir out]

Outputs saved to out directory (default: ./compare_centers_out):
 - per-cluster bar charts (local vs remote)
 - heatmap of absolute differences
 - PCA scatter of centers
 - JSON summary report
"""
import argparse
import ast
import json
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def load_centers(path):
    path = os.path.expanduser(path)
    if path.endswith('.npy') or path.endswith('.npz'):
        d = np.load(path, allow_pickle=True)
        if isinstance(d, np.lib.npyio.NpzFile):
            if 'centers' in d:
                return np.array(d['centers'])
            keys = list(d.keys())
            if keys:
                return np.array(d[keys[0]])
            raise ValueError(f'No arrays found in npz {path}')
        else:
            return np.array(d)
    # try plain numeric table
    try:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        pass
    # try python literal
    try:
        s = open(path, 'r', encoding='utf-8').read().strip()
        obj = ast.literal_eval(s)
        arr = np.array(obj, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception as e:
        raise ValueError(f'Cannot load centers from {path}: {e}')


def load_feats(path):
    if not path:
        return None
    with open(os.path.expanduser(path), 'r', encoding='utf-8') as fh:
        return json.load(fh)


def permute_match(a, b):
    # returns best permutation of rows in a to match b
    k = a.shape[0]
    best = None
    best_perm = None
    for perm in itertools.permutations(range(k)):
        perm = list(perm)
        diff = a[perm] - b
        sse = float((diff**2).sum())
        if best is None or sse < best:
            best = sse
            best_perm = perm
    matched_a = a[best_perm]
    diffs = matched_a - b
    return best_perm, matched_a, diffs, best


def align_by_features(a, a_feats, b, b_feats):
    common = [f for f in a_feats if f in b_feats]
    idx_a = [a_feats.index(f) for f in common]
    idx_b = [b_feats.index(f) for f in common]
    if not common:
        raise ValueError('No common features between supplied feature lists')
    return a[:, idx_a], b[:, idx_b], common


def try_auto_align(a, b):
    if a.shape[1] == b.shape[1]:
        return a, b, None, 'equal_shapes'
    if abs(a.shape[1] - b.shape[1]) == 1:
        candidates = []
        if a.shape[1] > b.shape[1]:
            a2 = a[:, 1:]
            if a2.shape[1] == b.shape[1]:
                _, ma, diffs, sse = permute_match(a2, b)
                candidates.append(('drop_first_local', a2, b, sse))
            a2 = a[:, :-1]
            if a2.shape[1] == b.shape[1]:
                _, ma, diffs, sse = permute_match(a2, b)
                candidates.append(('drop_last_local', a2, b, sse))
        if b.shape[1] > a.shape[1]:
            b2 = b[:, 1:]
            if b2.shape[1] == a.shape[1]:
                _, ma, diffs, sse = permute_match(a, b2)
                candidates.append(('drop_first_remote', a, b2, sse))
            b2 = b[:, :-1]
            if b2.shape[1] == a.shape[1]:
                _, ma, diffs, sse = permute_match(a, b2)
                candidates.append(('drop_last_remote', a, b2, sse))
        if not candidates:
            raise ValueError('Cannot auto-align: shapes differ and no simple drop yields same width')
        best = min(candidates, key=lambda t: t[3])
        reason, a_al, b_al, sse = best
        return a_al, b_al, reason, 'auto_dropped'
    else:
        minc = min(a.shape[1], b.shape[1])
        a2 = a[:, :minc]
        b2 = b[:, :minc]
        return a2, b2, f'truncate_to_{minc}', 'truncated_prefix'


def plot_per_cluster_bars(matched_a, b_al, feats, out_dir):
    k = matched_a.shape[0]
    os.makedirs(out_dir, exist_ok=True)
    for i in range(k):
        fig, ax = plt.subplots(figsize=(max(6, len(feats)*0.4), 4))
        x = np.arange(len(feats))
        width = 0.35
        ax.bar(x - width/2, matched_a[i], width, label='local')
        ax.bar(x + width/2, b_al[i], width, label='remote')
        ax.set_xticks(x)
        ax.set_xticklabels(feats, rotation=45, ha='right')
        ax.set_title(f'Cluster {i} centers (local vs remote)')
        ax.legend()
        plt.tight_layout()
        fp = os.path.join(out_dir, f'cluster_{i}_bars.png')
        fig.savefig(fp)
        plt.close(fig)


def plot_diff_heatmap(matched_a, b_al, feats, out_dir):
    diffs = np.abs(matched_a - b_al)
    fig, ax = plt.subplots(figsize=(max(6, len(feats)*0.4), max(3, matched_a.shape[0]*0.5)))
    sns.heatmap(diffs, annot=True, fmt='.3g', xticklabels=feats, yticklabels=[f'c{i}' for i in range(diffs.shape[0])], ax=ax)
    ax.set_title('Absolute differences (local vs remote)')
    plt.tight_layout()
    fp = os.path.join(out_dir, 'diff_heatmap.png')
    fig.savefig(fp)
    plt.close(fig)


def plot_pca_scatter(a_al, b_al, out_dir):
    X = np.vstack([a_al, b_al])
    labels = ['local']*a_al.shape[0] + ['remote']*b_al.shape[0]
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(x=Z[:a_al.shape[0],0], y=Z[:a_al.shape[0],1], label='local', marker='o', s=100)
    sns.scatterplot(x=Z[a_al.shape[0]:,0], y=Z[a_al.shape[0]:,1], label='remote', marker='X', s=100)
    for i in range(a_al.shape[0]):
        ax.text(Z[i,0], Z[i,1], f'L{i}', fontsize=9)
    for j in range(b_al.shape[0]):
        ax.text(Z[a_al.shape[0]+j,0], Z[a_al.shape[0]+j,1], f'R{j}', fontsize=9)
    ax.set_title('PCA of centers (local vs remote)')
    plt.tight_layout()
    fp = os.path.join(out_dir, 'centers_pca.png')
    fig.savefig(fp)
    plt.close(fig)


def summarize(matched_a, b_al, perm):
    diffs = matched_a - b_al
    abs_diff = np.abs(diffs)
    stats = {
        'max_abs_diff': float(np.max(abs_diff)),
        'mean_abs_diff': float(np.mean(abs_diff)),
        'per_feature_max_abs': list(map(float, abs_diff.max(axis=0))),
        'per_feature_mean_abs': list(map(float, abs_diff.mean(axis=0))),
        'per_cluster_norm': list(map(float, np.linalg.norm(abs_diff, axis=1)))
    }
    return stats


def parse_centers_string(s, expected_rows=None):
    """Parse a centers string that may be a Python literal, JSON, or space/bracket separated matrix.
    Returns a numpy array shaped (k, m) or (1, m) if single row.
    """
    s = s.strip()
    # strip surrounding quotes if present
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1]
    # normalize escape sequences and whitespace
    s = s.replace('\\n', ' ').replace('\n', ' ')
    s = ' '.join(s.split())

    # try python literal
    try:
        obj = ast.literal_eval(s)
        arr = np.array(obj, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        pass

    # try json
    try:
        obj = json.loads(s)
        arr = np.array(obj, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        pass

    # fallback: extract floats and split into rows by bracket boundaries
    import re
    inner = s
    if inner.startswith('[') and inner.endswith(']'):
        inner = inner[1:-1]
    # split rows by '][' or '] [' patterns
    row_chunks = re.split(r"\]\s*\[", inner)
    rows = []
    float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    for chunk in row_chunks:
        nums = float_re.findall(chunk)
        if nums:
            rows.append([float(x) for x in nums])
    if rows:
        maxc = max(len(r) for r in rows)
        mat = np.array([r + [np.nan] * (maxc - len(r)) for r in rows], dtype=float)
        return mat

    # final fallback: all floats
    floats = float_re.findall(s)
    if not floats:
        raise ValueError('No numeric values found in centers string')
    arr = np.array([float(x) for x in floats], dtype=float)
    if expected_rows is not None and arr.size % expected_rows == 0:
        arr = arr.reshape(expected_rows, -1)
        return arr
    return arr.reshape(1, -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--local', required=True)
    p.add_argument('--remote', help='Remote centers file (optional if --remote-assemble used)')
    p.add_argument('--remote-assemble', help='Path to dfanalyzer oAssemble.csv; use last row and --assemble-col to extract centers')
    p.add_argument('--assemble-col', type=int, default=2, help='1-based column index in oAssemble.csv to read (default: 2)')
    p.add_argument('--local-feats')
    p.add_argument('--remote-feats')
    p.add_argument('--out-dir', default='compare_centers_out')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    a = load_centers(args.local)
    # handle remote centers either from file or from oAssemble last-row column
    if args.remote_assemble:
        import csv
        assemble_path = os.path.expanduser(args.remote_assemble)
        if not os.path.exists(assemble_path):
            raise SystemExit(f'oAssemble file not found: {assemble_path}')
        with open(assemble_path, newline='', encoding='utf-8', errors='replace') as fh:
            reader = list(csv.reader(fh))
        if not reader:
            raise SystemExit('oAssemble.csv is empty')
        last = reader[-1]
        col_idx = max(1, args.assemble_col) - 1
        if col_idx >= len(last):
            raise SystemExit(f'assemble_col {args.assemble_col} out of range for last row with {len(last)} columns')
        raw = last[col_idx]
        try:
            b = parse_centers_string(raw, expected_rows=a.shape[0])
        except Exception as e:
            raise SystemExit(f'Failed to parse centers from oAssemble column: {e}')
    else:
        if not args.remote:
            raise SystemExit('Either --remote or --remote-assemble must be provided')
        b = load_centers(args.remote)
    if args.verbose:
        print('local centers shape:', a.shape)
        print('remote centers shape:', b.shape)

    if a.shape[0] != b.shape[0]:
        raise SystemExit(f'Different number of clusters: {a.shape[0]} vs {b.shape[0]}')

    a_feats = load_feats(args.local_feats) if args.local_feats else None
    b_feats = load_feats(args.remote_feats) if args.remote_feats else None

    if a_feats is not None and b_feats is not None:
        a_al, b_al, common = align_by_features(a, a_feats, b, b_feats)
        feats = common
        align_reason = 'feature_list_intersection'
    else:
        a_al, b_al, reason, method = try_auto_align(a, b)
        feats = [f'feat{i}' for i in range(a_al.shape[1])]
        align_reason = f'{method} ({reason})'
    perm, matched_a, diffs, sse = permute_match(a_al, b_al)
    stats = summarize(matched_a, b_al, perm)

    os.makedirs(args.out_dir, exist_ok=True)
    # plots
    plot_per_cluster_bars(matched_a, b_al, feats, args.out_dir)
    plot_diff_heatmap(matched_a, b_al, feats, args.out_dir)
    plot_pca_scatter(a_al, b_al, args.out_dir)

    report = {
        'local_input': args.local,
        'remote_input': args.remote,
        'align_reason': align_reason,
        'mapping_local_to_remote': perm,
        'shapes_after_alignment': [int(matched_a.shape[0]), int(matched_a.shape[1])],
        'stats': stats,
    }
    with open(os.path.join(args.out_dir, 'compare_report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)

    print('Wrote visualizations and report to', args.out_dir)


if __name__ == '__main__':
    main()
