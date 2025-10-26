import argparse, numpy as np, pandas as pd
from features.features import normalize_df, featurize_pair, label_heuristic

def sample_pairs(df, n_pairs=30000, level_low=50, level_high=80, seed=42):
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(df), size=n_pairs)
    idx_b = rng.integers(0, len(df), size=n_pairs)
    # avoid identical pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    n = len(idx_a)
    levels_a = rng.integers(level_low, level_high+1, size=n)
    levels_b = rng.integers(level_low, level_high+1, size=n)
    rows = []
    for i in range(n):
        a = df.iloc[idx_a[i]]; b = df.iloc[idx_b[i]]
        la, lb = int(levels_a[i]), int(levels_b[i])
        feat = featurize_pair(a, b, la, lb)
        y = label_heuristic(a, b, la, lb)
        rows.append({
            "name_a": a["Name"], "name_b": b["Name"], "level_a": la, "level_b": lb, "y": y, **feat
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokemon_csv", default="data/pokemon.csv")
    ap.add_argument("--out", default="model/train_pairs.parquet")
    ap.add_argument("--pairs", type=int, default=30000)
    args = ap.parse_args()

    raw = pd.read_csv(args.pokemon_csv)
    df = normalize_df(raw).dropna(subset=["Name","Type 1"])
    df = df.reset_index(drop=True)

    pairs = sample_pairs(df, n_pairs=args.pairs)
    pairs.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} with {len(pairs)} rows")
