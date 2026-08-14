"""
Run EXP_008: Label Encoding Redundancy
Active base: EXP_004
"""
import json, subprocess, re, sys, os

NOTEBOOK = 'Traffic_Demand_Prediction.ipynb'
SCRATCH = 'scratch/exp_008_metrics.txt'
os.makedirs('scratch', exist_ok=True)

def reset_to_exp004():
    """Re-apply EXP_004 base patch (24 features)"""
    import patch_nb
    patch_nb.patch_nb()
    with open(NOTEBOOK, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    src = nb['cells'][10]['source']
    # Clean up any injected drops
    src = [l for l in src if "drop(columns=['geohash_te" not in l and "drop(columns=['geohash_freq" not in l and "drop(columns=['geohash_enc'" not in l]
    nb['cells'][10]['source'] = src
    with open(NOTEBOOK, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("[reset_to_exp004] Notebook reset to EXP_004 (24 features).")

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
print("STEP 1: EXP_008 - Drop Spatial Label Encodings")
print("="*60)
# Ensure we start from a clean state
reset_to_exp004()

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    nb = json.load(f)
src = nb['cells'][10]['source']
new_src = []
drop_cols = "'geohash_enc', 'geohash_prefix_3_enc', 'geohash_prefix_4_enc', 'geohash_prefix_5_enc'"
for line in src:
    new_src.append(line)
    if line.startswith("test_df   = full_df[full_df['is_train'] == 0]"):
        new_src.append(f"train_df = train_df.drop(columns=[{drop_cols}], errors='ignore')\n")
        new_src.append(f"test_df  = test_df.drop(columns=[{drop_cols}], errors='ignore')\n")
nb['cells'][10]['source'] = new_src
with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Running EXP_008...")
run_notebook()
kf8, gkf8, gap8 = extract_metrics()
print(f"EXP_008 Metrics: KFold={kf8} GKF={gkf8} Gap={gap8}")

# Accept rule:
# ACCEPT if GroupKFold >= 0.9171 AND gap does not widen (i.e. <= 0.0245)
# REJECT if GroupKFold drops by more than 0.001 (i.e. < 0.9161)
decision = "REJECT"
if gkf8 is not None and gkf8 >= 0.9171 and gap8 is not None and gap8 <= 0.0245:
    decision = "ACCEPT"
elif gkf8 is not None and gkf8 < 0.9161:
    decision = "REJECT"
else:
    # Gray area: e.g. GKF=0.9165 but gap widened. The prompt says REJECT if it drops by more than 0.001,
    # but also ACCEPT if >= 0.9171 and gap doesn't widen. I'll strictly REJECT if it doesn't meet ACCEPT.
    decision = "REJECT"

with open(SCRATCH, 'w', encoding='utf-8') as f:
    f.write(f"EXP_008: KFold={kf8} GKF={gkf8} Gap={gap8} Decision={decision}\n")

if decision == "REJECT":
    print("\nEXP_008 REJECTED. Reverting to EXP_004 config...")
    reset_to_exp004()
    run_notebook() # Regen for EXP_004 base

print("\n=== SCRIPT FINISHED ===")
