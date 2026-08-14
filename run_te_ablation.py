"""
TE Type Ablation Runner — EXP_005 and EXP_006
Base: EXP_004 (encode_cols=['geohash'], geohash_te_mean + geohash_te_median)

EXP_005: drop geohash_te_median  → keep smoothed mean only
EXP_006: drop geohash_te_mean    → keep median only
"""
import json, subprocess, re, sys, os, datetime

NOTEBOOK = 'Traffic_Demand_Prediction.ipynb'
SCRATCH   = 'scratch/exp_005_006_metrics.txt'
os.makedirs('scratch', exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def reset_to_exp004():
    """Re-apply EXP_004 base patch (encode_cols=['geohash'])."""
    import patch_nb
    patch_nb.patch_nb()
    print("[reset_to_exp004] patch_nb done.")


def inject_drop(col_to_drop: str):
    """
    After patch_nb has been applied, open the notebook and inject a drop line
    immediately after the test_df assignment in Cell 10.
    Only the specified column is dropped; nothing else changes.
    """
    with open(NOTEBOOK, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    src = nb['cells'][10]['source']

    # Remove any leftover drops from a previous run of this function
    src = [l for l in src
           if "drop(columns=['geohash_te_median']" not in l
           and "drop(columns=['geohash_te_mean']" not in l]

    new_src = []
    for line in src:
        new_src.append(line)
        if line.startswith("test_df   = full_df[full_df['is_train'] == 0]"):
            new_src.append(
                f"train_df = train_df.drop(columns=['{col_to_drop}'], errors='ignore')\n"
            )
            new_src.append(
                f"test_df  = test_df.drop(columns=['{col_to_drop}'], errors='ignore')\n"
            )

    nb['cells'][10]['source'] = new_src
    with open(NOTEBOOK, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"[inject_drop] Dropped '{col_to_drop}' from train_df/test_df.")


def run_notebook(timeout=700):
    print("[run_notebook] Executing notebook …")
    result = subprocess.run(
        ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
         f'--ExecutePreprocessor.timeout={timeout}',
         '--inplace', NOTEBOOK],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    print(f"[run_notebook] Exit code: {result.returncode}")
    if result.stderr:
        for line in result.stderr.splitlines()[-10:]:
            print("  STDERR:", line)
    return result.returncode


def extract_metrics():
    """Pull KFold ensemble R², GroupKFold R², and gap from notebook cell outputs."""
    with open(NOTEBOOK, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    kfold_r2 = gkf_r2 = gap = None

    for cell in nb['cells']:
        for out in cell.get('outputs', []):
            text = ''.join(out.get('text', []))
            if not text:
                continue
            # Ensemble R² line e.g. "[36% LGBM + 64% CB Ensemble] R²: 0.9417"
            m = re.search(r'Ensemble\].*?R.:\s*([\d.]+)', text)
            if m:
                kfold_r2 = float(m.group(1))
            # GroupKFold line e.g. "[GroupKFold LGBM]        R²: 0.9171"
            m = re.search(r'GroupKFold.*?R.:\s*([\d.]+)', text)
            if m:
                gkf_r2 = float(m.group(1))
            # Gap line e.g. "KFold → GroupKFold gap: 0.0245"
            m = re.search(r'gap:\s*([\d.]+)', text, re.IGNORECASE)
            if m:
                gap = float(m.group(1))

    return kfold_r2, gkf_r2, gap


def log_result(exp_id, change, kfold_r2, gkf_r2, gap):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = (f"{ts} | {exp_id} | {change} | "
            f"KFold={kfold_r2:.4f} | GKF={gkf_r2:.4f} | Gap={gap:.4f}\n")
    with open(SCRATCH, 'a', encoding='utf-8') as f:
        f.write(line)
    print(f"[log] {line.strip()}")


# ── EXP_005 ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXP_005: drop geohash_te_median (keep smoothed mean only)")
print("="*60)

reset_to_exp004()
inject_drop('geohash_te_median')
rc = run_notebook()
kf5, gkf5, gap5 = extract_metrics()
print(f"\nEXP_005 → KFold R²={kf5}  GKF R²={gkf5}  Gap={gap5}")
log_result('EXP_005', 'drop geohash_te_median (mean only)', kf5, gkf5, gap5)


# ── EXP_006 ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXP_006: drop geohash_te_mean (keep median only)")
print("="*60)

reset_to_exp004()
inject_drop('geohash_te_mean')
rc = run_notebook()
kf6, gkf6, gap6 = extract_metrics()
print(f"\nEXP_006 → KFold R²={kf6}  GKF R²={gkf6}  Gap={gap6}")
log_result('EXP_006', 'drop geohash_te_mean (median only)', kf6, gkf6, gap6)


# ── winner selection ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("WINNER SELECTION")
print("="*60)
print(f"EXP_005  GKF={gkf5:.4f}  (smoothed mean only)")
print(f"EXP_006  GKF={gkf6:.4f}  (median only)")

if gkf5 >= gkf6 - 0.001:
    winner = 'EXP_005'
    drop_for_winner = 'geohash_te_median'
    winner_kf, winner_gkf, winner_gap = kf5, gkf5, gap5
    winner_desc = 'smoothed mean only'
else:
    winner = 'EXP_006'
    drop_for_winner = 'geohash_te_mean'
    winner_kf, winner_gkf, winner_gap = kf6, gkf6, gap6
    winner_desc = 'median only'

print(f"\n>>> WINNER: {winner} ({winner_desc})")
print(f"    KFold={winner_kf:.4f}  GKF={winner_gkf:.4f}  Gap={winner_gap:.4f}")

# Leave notebook in winner config
print(f"\n[final] Setting notebook to {winner} config ({drop_for_winner} dropped) …")
reset_to_exp004()
inject_drop(drop_for_winner)
print("[final] Notebook set to winner config. Ready for artifact sync.")

# Write winner summary to scratch
with open(SCRATCH, 'a', encoding='utf-8') as f:
    f.write(f"\nWINNER: {winner} | {winner_desc} | KFold={winner_kf:.4f} | "
            f"GKF={winner_gkf:.4f} | Gap={winner_gap:.4f}\n")
    f.write(f"Drop for winner: {drop_for_winner}\n")

print("\n=== TE ABLATION BLOCK DONE — ready for artifact sync ===")
