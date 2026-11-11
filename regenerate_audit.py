"""One-shot: regenerate the DEPLOYED-model fairness audit, verify it, save results, optionally push.
Run from the repo root on branch `newbranch` (the PC that has TestDb/DATASET and the real .pth files):

    .venv\\Scripts\\python.exe regenerate_audit.py            # run + verify + save results
    .venv\\Scripts\\python.exe regenerate_audit.py --push     # ...and commit + push to newbranch

It changes NOTHING by hand: models/fairness_report.json is produced by fairness.py itself.
"""
import argparse, json, os, shutil, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="TestDb/DATASET")
ap.add_argument("--push", action="store_true")
a = ap.parse_args()

OUT = "audit_results"; os.makedirs(OUT, exist_ok=True)
def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str))

# 1. commit hash
commit = sh(["git", "rev-parse", "HEAD"]).stdout.strip()
branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
open(f"{OUT}/commit_hash.txt", "w").write(f"{commit}\n{branch}\n")
print("commit", commit, "branch", branch)

# 2. backup old report
if os.path.exists("models/fairness_report.json") and not os.path.exists("models/fairness_report_OLD.json"):
    shutil.copy("models/fairness_report.json", "models/fairness_report_OLD.json")

# 3. confirm ensemble json is untouched / single model
ens = json.load(open("models/emotion_ensemble.json"))
assert ens["models"] == ["models/emotion_r50_s7_cw03.pth"], ens["models"]
assert ens["calibration_method"] == "logit_adjustment (tau=0.50)", ens["calibration_method"]
exp = [-0.0219, 0.7401, 0.2717, -0.676, -0.2366, 0.2802, -0.3575]
assert all(abs(x - y) < 1e-4 for x, y in zip(ens["class_bias"], exp)), ens["class_bias"]
print("emotion_ensemble.json OK (single model, tau 0.50, class_bias matches)")

# 4. regenerate the audit with fairness.py (deployed model + baseline only)
res = subprocess.run([sys.executable, "fairness.py", "--data_dir", a.data_dir,
                      "--extra_models", "models/emotion_r50_nofair.pth"], capture_output=True, text=True)
open(f"{OUT}/fairness_run_log.txt", "w", encoding="utf-8").write(res.stdout + "\n--- stderr ---\n" + res.stderr)
print(res.stdout[-3000:])
if res.returncode != 0:
    print(res.stderr[-2000:]); sys.exit("fairness.py failed - see audit_results/fairness_run_log.txt")

# 5. verify
r = json.load(open("models/fairness_report.json")); f = r["final"]
lines = ["name: " + f["name"]]; bad = 0
for row in f["per_class"]:
    tp = row["recall"] * row["support"]; ok = abs(tp - round(tp)) < 0.01; bad += (not ok)
    lines.append(f'{row["class"]:10s} support={row["support"]:4d} recall={row["recall"]:.4f} tp={tp:.2f} {"OK" if ok else "NOT INTEGER"}')
lines += ["summary: " + json.dumps(f["summary"]), f"non-integer rows: {bad} (must be 0)"]
if "baseline" in r: lines.append("baseline summary: " + json.dumps(r["baseline"]["summary"]))
open(f"{OUT}/fairness_check.txt", "w").write("\n".join(lines)); print("\n".join(lines))
acc = f["summary"].get("accuracy"); print("deployed accuracy", acc, "(expected ~0.8902)")

# 6. checkpoint training settings
import torch
ck = torch.load("models/emotion_r50_s7_cw03.pth", map_location="cpu")
info = {k: v for k, v in ck.items() if k != "state_dict"}
open(f"{OUT}/ckpt_info.txt", "w").write(str(info)); print("ckpt info:", info)

ok = bad == 0 and acc is not None and abs(acc - 0.8902) < 0.002
print("\nSANITY:", "PASS" if ok else "CHECK NEEDED (do not edit by hand; send audit_results/ to Vaishnav)")

# 7. optional push (only if sanity passed)
if a.push:
    if not ok: sys.exit("sanity not passed - not pushing")
    sh(["git", "add", "models/fairness_report.json", "models/fairness_report.png", "audit_results"])
    print(sh(["git", "commit", "-m", "Regenerate deployed-model fairness audit (single model s7, tau 0.50) + audit_results"]).stdout)
    print(sh(["git", "push", "origin", branch]).stdout)

