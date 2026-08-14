"""
REVERT to EXP_004 config & Run EXP_007
"""
import json, subprocess, re, sys, os, datetime
import patch_nb

NOTEBOOK = 'Traffic_Demand_Prediction.ipynb'
SCRATCH = 'scratch/exp_007_metrics.txt'
os.makedirs('scratch', exist_ok=True)

def reset_to_exp004():
    """Re-apply EXP_004 base patch (encode_cols=['geohash'], no drops)"""
    patch_nb.patch_nb()
    with open(NOTEBOOK, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    src = nb['cells'][10]['source']
    # Clean up any injected drops
    src = [l for l in src if "drop(columns=['geohash_te" not in l and "drop(columns=['geohash_freq" not in l]
    nb['cells'][10]['source'] = src
    with open(NOTEBOOK, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("[reset_to_exp004] Notebook reset to EXP_004 (24 features: mean+median TE retained).")

def run_notebook(timeout=700):
    print("[run_notebook] Executing notebook...")
    result = subprocess.run(
        ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
         f'--ExecutePreprocessor.timeout={timeout}', '--inplace', NOTEBOOK],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    print(f"[run_notebook] Exit code: {result.returncode}")
    return result.returncode

def extract_metrics():
    with open(NOTEBOOK, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    kfold_r2 = gkf_r2 = gap = None
    for cell in nb['cells']:
        for out in cell.get('outputs', []):
            text = ''.join(out.get('text', []))
            m = re.search(r'Ensemble\].*?R.:\s*([\d.]+)', text)
            if m: kfold_r2 = float(m.group(1))
            m = re.search(r'GroupKFold.*?R.:\s*([\d.]+)', text)
            if m: gkf_r2 = float(m.group(1))
            m = re.search(r'gap:\s*([\d.]+)', text, re.IGNORECASE)
            if m: gap = float(m.group(1))
    return kfold_r2, gkf_r2, gap

print("\n" + "="*60)
print("STEP 1: REVERT TO EXP_004")
print("="*60)
reset_to_exp004()
print("Executing notebook to regenerate base submission_final.csv...")
run_notebook()
kf4, gkf4, gap4 = extract_metrics()
print(f"EXP_004 Base Metrics: KFold={kf4} GKF={gkf4} Gap={gap4}")

print("\n" + "="*60)
print("STEP 2: EXP_007 - Drop Frequency Bins")
print("="*60)
with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    nb = json.load(f)
src = nb['cells'][10]['source']
new_src = []
for line in src:
    new_src.append(line)
    if line.startswith("test_df   = full_df[full_df['is_train'] == 0]"):
        new_src.append("train_df = train_df.drop(columns=['geohash_freq', 'geohash_freq_bin_enc'], errors='ignore')\n")
        new_src.append("test_df  = test_df.drop(columns=['geohash_freq', 'geohash_freq_bin_enc'], errors='ignore')\n")
nb['cells'][10]['source'] = new_src
with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Running EXP_007...")
run_notebook()
kf7, gkf7, gap7 = extract_metrics()
print(f"EXP_007 Metrics: KFold={kf7} GKF={gkf7} Gap={gap7}")

# Determine winner for Step 3
if gkf7 is not None and gkf7 >= 0.9171 - 0.001:  # GroupKFold holds or improves
    decision = "ACCEPT"
else:
    decision = "REJECT"

with open(SCRATCH, 'w', encoding='utf-8') as f:
    f.write(f"EXP_007: KFold={kf7} GKF={gkf7} Gap={gap7} Decision={decision}\n")

if decision == "REJECT":
    print("\nEXP_007 REJECTED. Reverting to EXP_004 config...")
    reset_to_exp004()
    run_notebook() # Regen for EXP_004

print("\n=== SCRIPT FINISHED ===")
