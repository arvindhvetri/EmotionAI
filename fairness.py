"""
Fairness metrics for the FA&EFER emotion engine.

Two uses
--------
1. Library (imported by app.py): per-class / per-group metrics, disparity summaries and a
   per-prediction reliability context for the web app and the batch PDF report.
2. Command line: a full fairness audit of the trained emotion models on the RAF-DB test set.

       venv\\Scripts\\python.exe fairness.py --data_dir TestDb/DATASET
       # optional demographic audit, if you have the RAF-DB manual attribute files:
       venv\\Scripts\\python.exe fairness.py --data_dir TestDb/DATASET --attributes "<RAF-DB>/basic/Annotation/manual"

   Writes models/fairness_report.json (read by app.py) and models/fairness_report.png.

What "fairness" means here
--------------------------
RAF-DB as used in this project has no demographic labels, so the default audit treats each
emotion class as a group and checks whether the model serves every class equally well
(an equal-opportunity view: does each true class get the same chance of being recognised?).

* recall (true-positive rate) per class, and the gap / ratio between best and worst class
  - recall ratio = worst recall / best recall; >= 0.80 passes the "four-fifths" rule of thumb
* false-negative and false-positive rates per class
* balanced accuracy (mean recall) vs accuracy: a large difference means the headline
  accuracy is carried by the majority classes
* prediction-rate bias: share of predictions per class minus true share (over/under-prediction)

If demographic attributes are supplied (--attributes), the same model is also audited per
gender / race / age group: accuracy, macro-F1 and the equalized-odds gap (largest difference
in per-class recall between groups).
"""
import argparse
import json
import math
import os

import numpy as np

EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
# RAF-DB basic-emotion training counts (folders 1..7) - used for the training share of each class
TRAIN_COUNTS = {"Surprise": 1290, "Fear": 281, "Disgust": 717, "Happiness": 4772,
                "Sadness": 1982, "Anger": 705, "Neutral": 2524}
MINORITY_SHARE = 0.07          # classes with < 7% of training images are flagged as minority classes
FOUR_FIFTHS = 0.80


# --------------------------------------------------------------------------
# Core metrics (numpy only - safe to import in the web app)
# --------------------------------------------------------------------------
def per_class_metrics(y_true, y_pred, names=EMOTIONS):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n, total_train = len(y_true), sum(TRAIN_COUNTS.values())
    rows = []
    for k, name in enumerate(names):
        t, p = y_true == k, y_pred == k
        tp, fn, fp = int((t & p).sum()), int((t & ~p).sum()), int((~t & p).sum())
        tn = n - tp - fn - fp
        support = int(t.sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support if support else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if support and prec + rec else 0.0
        rows.append({
            "class": name, "support": support,
            "train_share": round(TRAIN_COUNTS.get(name, 0) / total_train, 4),
            "true_share": round(support / n, 4) if n else 0.0,
            "pred_share": round(int(p.sum()) / n, 4) if n else 0.0,
            "precision": round(prec, 4), "recall": None if math.isnan(rec) else round(rec, 4),
            "f1": round(f1, 4), "fnr": None if math.isnan(rec) else round(1 - rec, 4),
            "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        })
    return rows


def disparity_summary(rows, y_true=None, y_pred=None):
    valid = [r for r in rows if r["support"] > 0 and r["recall"] is not None]
    if not valid:
        return {}
    rec = {r["class"]: r["recall"] for r in valid}
    f1 = {r["class"]: r["f1"] for r in valid}
    worst, best = min(rec, key=rec.get), max(rec, key=rec.get)
    s = {
        "classes_evaluated": len(valid),
        "balanced_accuracy": round(float(np.mean(list(rec.values()))), 4),
        "macro_f1": round(float(np.mean(list(f1.values()))), 4),
        "worst_recall_class": worst, "worst_recall": rec[worst],
        "best_recall_class": best, "best_recall": rec[best],
        "recall_gap": round(rec[best] - rec[worst], 4),
        "recall_ratio": round(rec[worst] / rec[best], 4) if rec[best] else 0.0,
        "recall_std": round(float(np.std(list(rec.values()))), 4),
        "worst_f1_class": min(f1, key=f1.get), "worst_f1": min(f1.values()),
        "f1_gap": round(max(f1.values()) - min(f1.values()), 4),
        "fpr_gap": round(max(r["fpr"] for r in valid) - min(r["fpr"] for r in valid), 4),
        "max_prediction_bias": max(((r["class"], round(r["pred_share"] - r["true_share"], 4)) for r in valid),
                                   key=lambda x: abs(x[1])),
    }
    s["passes_four_fifths"] = s["recall_ratio"] >= FOUR_FIFTHS
    if y_true is not None and y_pred is not None:
        acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
        s["accuracy"] = round(acc, 4)
        s["accuracy_minus_balanced"] = round(acc - s["balanced_accuracy"], 4)
    minority = [r for r in valid if r["train_share"] < MINORITY_SHARE]
    majority = [r for r in valid if r["train_share"] >= MINORITY_SHARE]
    if minority and majority:
        s["minority_classes"] = [r["class"] for r in minority]
        s["minority_mean_recall"] = round(float(np.mean([r["recall"] for r in minority])), 4)
        s["majority_mean_recall"] = round(float(np.mean([r["recall"] for r in majority])), 4)
        s["minority_recall_gap"] = round(s["majority_mean_recall"] - s["minority_mean_recall"], 4)
    return s


def group_metrics(y_true, y_pred, groups, names=EMOTIONS):
    """Accuracy, macro-F1 and per-class recall for each demographic group, plus the gaps."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    out, recalls = {}, {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        rows = per_class_metrics(y_true[m], y_pred[m], names)
        valid = [r for r in rows if r["support"] > 0]
        out[g] = {"n": int(m.sum()), "accuracy": round(float((y_true[m] == y_pred[m]).mean()), 4),
                  "macro_f1": round(float(np.mean([r["f1"] for r in valid])), 4) if valid else 0.0}
        recalls[g] = {r["class"]: r["recall"] for r in rows if r["support"] >= 5}
    accs = [v["accuracy"] for v in out.values()]
    f1s = [v["macro_f1"] for v in out.values()]
    eo = 0.0
    for c in names:
        vals = [recalls[g][c] for g in recalls if c in recalls[g]]
        if len(vals) >= 2:
            eo = max(eo, max(vals) - min(vals))
    return {"groups": out, "accuracy_gap": round(max(accs) - min(accs), 4) if accs else 0.0,
            "macro_f1_gap": round(max(f1s) - min(f1s), 4) if f1s else 0.0,
            "equalized_odds_gap": round(eo, 4)}


COMPARE_KEYS = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"),
                ("worst_recall", "Worst-class recall"), ("recall_gap", "Recall gap (best - worst)"),
                ("recall_ratio", "Recall ratio (worst / best)"), ("minority_mean_recall", "Minority-class mean recall")]


def compare_with_without(without, with_):
    """Side-by-side of a system WITHOUT fairness mitigation and one WITH it.
    Each argument is {"per_class": [...], "summary": {...}} as produced by per_class_metrics/disparity_summary."""
    metrics = {}
    for k, label in COMPARE_KEYS:
        a, b = without["summary"].get(k), with_["summary"].get(k)
        if a is not None and b is not None:
            metrics[k] = {"label": label, "without": a, "with": b, "change": round(b - a, 4)}
    per_class = {}
    for ra, rb in zip(without["per_class"], with_["per_class"]):
        if ra["recall"] is not None and rb["recall"] is not None:
            per_class[ra["class"]] = {"without": ra["recall"], "with": rb["recall"],
                                      "change": round(rb["recall"] - ra["recall"], 4), "train_share": ra["train_share"]}
    return {"metrics": metrics, "per_class_recall": per_class,
            "four_fifths": {"without": without["summary"].get("passes_four_fifths"),
                            "with": with_["summary"].get("passes_four_fifths")}}


# --------------------------------------------------------------------------
# Per-prediction context (used by /predict and the batch report)
# --------------------------------------------------------------------------
def load_report(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def prediction_context(predicted, probs, report=None, facs_emotion=None, names=EMOTIONS):
    """Reliability and fairness context for one prediction."""
    p = np.asarray(probs, dtype=float)
    order = np.argsort(p)[::-1]
    conf, margin = float(p[order[0]]), float(p[order[0]] - p[order[1]])
    entropy = float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p)))
    total_train = sum(TRAIN_COUNTS.values())
    share = TRAIN_COUNTS.get(predicted, 0) / total_train
    ctx = {
        "confidence": round(conf, 4), "margin": round(margin, 4), "uncertainty": round(entropy, 4),
        "runner_up": names[int(order[1])],
        "train_share": round(share, 4), "minority_class": share < MINORITY_SHARE,
    }
    cls = None
    if report and "final" in report:
        cls = next((r for r in report["final"]["per_class"] if r["class"] == predicted), None)
    if cls:
        ctx.update({"class_test_recall": cls["recall"], "class_test_precision": cls["precision"],
                    "class_test_f1": cls["f1"], "class_test_support": cls["support"]})
    if facs_emotion is not None:
        ctx["facs_rule_emotion"] = facs_emotion
        ctx["facs_agreement"] = facs_emotion == predicted
    class_f1 = cls["f1"] if cls else None
    if conf < 0.5 or (class_f1 is not None and class_f1 < 0.78):
        band = "Low"
    elif conf >= 0.7 and (class_f1 is None or class_f1 >= 0.85):
        band = "High"
    else:
        band = "Medium"
    ctx["reliability"] = band
    ctx["needs_review"] = band == "Low" or (margin < 0.15)
    notes = []
    if ctx["minority_class"]:
        notes.append(f"{predicted} is a minority class ({share*100:.1f}% of training images)")
    if cls:
        notes.append(f"on the test set the model recognises {cls['recall']*100:.1f}% of true {predicted} faces "
                     f"and {cls['precision']*100:.1f}% of its {predicted} predictions are correct")
    if margin < 0.15:
        notes.append(f"close call with {ctx['runner_up']}")
    if facs_emotion is not None and not ctx["facs_agreement"]:
        notes.append(f"the FACS rules suggest {facs_emotion}")
    text = "; ".join(notes)
    ctx["note"] = (text[0].upper() + text[1:] + ".") if notes else "Confident prediction of a well-recognised class."
    return ctx


# --------------------------------------------------------------------------
# Optional demographic attributes (RAF-DB manual annotations)
# --------------------------------------------------------------------------
GENDER = {0: "male", 1: "female", 2: "unsure"}
RACE = {0: "Caucasian", 1: "African-American", 2: "Asian"}
AGE = {0: "0-3", 1: "4-19", 2: "20-39", 3: "40-69", 4: "70+"}


def load_rafdb_attributes(folder):
    """Reads RAF-DB manual attribute files (<image>_manu_attri.txt). The last three numeric lines are
    taken as gender, race and age codes. Check one file by hand: if your copy uses a different
    layout, adjust this function."""
    attrs = {}
    for f in os.listdir(folder):
        if not f.endswith("_manu_attri.txt"):
            continue
        with open(os.path.join(folder, f)) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        codes = [l for l in lines if l.isdigit()][-3:]
        if len(codes) == 3:
            g, r, a = map(int, codes)
            attrs[f.replace("_manu_attri.txt", "")] = {"gender": GENDER.get(g, str(g)),
                                                        "race": RACE.get(r, str(r)), "age": AGE.get(a, str(a))}
    return attrs


# --------------------------------------------------------------------------
# Command-line audit
# --------------------------------------------------------------------------
def _evaluate(name, probs, y, bias=None):
    logp = np.log(probs + 1e-9) + (0 if bias is None else np.asarray(bias))
    pred = logp.argmax(1)
    rows = per_class_metrics(y, pred)
    return {"name": name, "per_class": rows, "summary": disparity_summary(rows, y, pred)}, pred


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configs = [report["baseline"], report["final"]] if "baseline" in report else report["comparison"]
    x = np.arange(len(EMOTIONS)); w = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, c in enumerate(configs):
        ax.bar(x + i * w - 0.4 + w / 2, [r["recall"] or 0 for r in c["per_class"]], w, label=c["name"])
    ax.axhline(report["final"]["summary"]["best_recall"] * FOUR_FIFTHS, color="grey", ls="--", lw=1,
               label="4/5 of best recall (final model)")
    ax.set_xticks(x); ax.set_xticklabels(EMOTIONS); ax.set_ylim(0, 1.05); ax.set_ylabel("Recall (true-positive rate)")
    ax.set_title("Per-class recall on the RAF-DB test set" + (" - with vs without fairness" if "baseline" in report
                 else " - fairness across emotion classes"))
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def main():
    import torch
    from torch.utils.data import DataLoader
    from ensemble_emotion import predict_probs
    from train_emotion_resnet50_vggface2 import ImageList, VGGFace2ResNet50, build_transforms, list_images

    ap = argparse.ArgumentParser(description="Fairness audit of the emotion models on the RAF-DB test set")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ensemble_json", default="models/emotion_ensemble.json")
    ap.add_argument("--extra_models", nargs="*",
                    default=["models/emotion_r50_nofair.pth", "models/best_resnet50_vggface2_emotion.pth"],
                    help="single models to include in the comparison (emotion_r50_nofair.pth = baseline without "
                         "fairness, trained by train_nofair.bat)")
    ap.add_argument("--attributes", default=None, help="folder with RAF-DB *_manu_attri.txt files (optional)")
    ap.add_argument("--out_json", default="models/fairness_report.json")
    ap.add_argument("--out_png", default="models/fairness_report.png")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ens = load_report(a.ensemble_json) or {}
    # ensemble paths are stored as models/<file>; fall back to the folder of the JSON (e.g. Google Drive on Colab)
    ens_models = [m if os.path.exists(m) else os.path.join(os.path.dirname(os.path.abspath(a.ensemble_json)),
                                                          os.path.basename(m)) for m in ens.get("models", [])]
    bias = ens.get("class_bias", [0.0] * 7)
    paths = list(dict.fromkeys([m for m in a.extra_models + ens_models if os.path.exists(m)]))
    if not paths:
        raise SystemExit("No model checkpoints found.")

    _, tf = build_transforms(ens.get("img_size", 224))
    te = ImageList(list_images(os.path.join(a.data_dir, "test")), tf)
    y = np.array([lab for _, lab in te.items])
    loader = DataLoader(te, a.batch_size, num_workers=a.workers)
    probs = {}
    for pth in paths:
        net = VGGFace2ResNet50(); net.load_state_dict(torch.load(pth, map_location=device)["state_dict"]); net.to(device)
        probs[pth] = predict_probs(net, loader, device)
        print("[eval]", os.path.basename(pth)); del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison, baseline = [], None
    for pth in paths:
        ck = torch.load(pth, map_location="cpu")
        stem = os.path.splitext(os.path.basename(pth))[0]
        if ck.get("fairness_mitigation") is False:
            entry = _evaluate(f"WITHOUT fairness: {stem} (no weights, acc. selection)", probs[pth], y)[0]
            entry["role"] = "without_fairness"
            baseline = baseline or entry
        else:
            cwp = ck.get("class_weight_power")
            entry = _evaluate(f"with fairness: {stem}" + (f" (class weight p={cwp})" if cwp is not None else ""),
                              probs[pth], y)[0]
        comparison.append(entry)
    final_pred = None
    if ens_models and all(m in probs for m in ens_models):
        pe = np.mean([probs[m] for m in ens_models], 0)
        comparison.append(_evaluate("ensemble (no calibration)", pe, y)[0])
        final, final_pred = _evaluate("with fairness: ensemble + per-class calibration (DEPLOYED)", pe, y, bias)
        comparison.append(final)
    else:
        final, final_pred = comparison[-1], None
        final_pred = (np.log(probs[paths[-1]] + 1e-9)).argmax(1)

    report = {"dataset": "RAF-DB test", "n_test": int(len(y)), "final": final, "comparison": comparison,
              "definition": "Emotion classes treated as groups (no demographic labels). Recall ratio >= 0.80 = four-fifths rule."}
    if baseline:
        report["baseline"] = baseline
        report["with_vs_without"] = compare_with_without(baseline, final)
    else:
        print("[note] no baseline without fairness found (models/emotion_r50_nofair.pth) - run train_nofair.bat "
              "to add the with-vs-without comparison")

    if a.attributes:
        attrs = load_rafdb_attributes(a.attributes)
        keys = [os.path.splitext(os.path.basename(p))[0].replace("_aligned", "") for p, _ in te.items]
        have = np.array([k in attrs for k in keys])
        print(f"[attributes] matched {have.sum()} of {len(keys)} test images")
        if have.sum():
            report["demographic"] = {}
            for att in ["gender", "race", "age"]:
                g = np.array([attrs[k][att] for k, h in zip(keys, have) if h])
                report["demographic"][att] = group_metrics(y[have], final_pred[have], g)

    with open(a.out_json, "w") as f:
        json.dump(report, f, indent=2)
    _plot(report, a.out_png)

    print("\n================ FAIRNESS AUDIT (RAF-DB test, %d images) ================" % len(y))
    print(f"{'configuration':58s} {'acc':>6s} {'bal.acc':>7s} {'worst rec':>16s} {'rec gap':>7s} {'ratio':>6s} 4/5")
    for c in comparison:
        s = c["summary"]
        print(f"{c['name'][:58]:58s} {s['accuracy']*100:5.2f}% {s['balanced_accuracy']*100:6.2f}% "
              f"{s['worst_recall_class']:>9s} {s['worst_recall']*100:5.1f}% {s['recall_gap']*100:6.1f} "
              f"{s['recall_ratio']:6.3f} {'yes' if s['passes_four_fifths'] else 'NO'}")
    print("\nDeployed model, per class:")
    print(f"{'class':10s} {'train%':>7s} {'support':>7s} {'prec':>6s} {'recall':>6s} {'f1':>6s} {'FNR':>6s} {'pred-true share':>15s}")
    for r in final["per_class"]:
        print(f"{r['class']:10s} {r['train_share']*100:6.1f}% {r['support']:7d} {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['f1']:6.3f} {r['fnr']:6.3f} {(r['pred_share']-r['true_share'])*100:+14.1f}%")
    s = final["summary"]
    if "minority_recall_gap" in s:
        print(f"\nMinority classes {s['minority_classes']}: mean recall {s['minority_mean_recall']*100:.1f}% "
              f"vs {s['majority_mean_recall']*100:.1f}% for the others (gap {s['minority_recall_gap']*100:.1f} points)")
    print(f"Accuracy {s['accuracy']*100:.2f}% vs balanced accuracy {s['balanced_accuracy']*100:.2f}% "
          f"(difference {s['accuracy_minus_balanced']*100:+.2f} points)")
    for att, gm in report.get("demographic", {}).items():
        print(f"\n[{att}] accuracy gap {gm['accuracy_gap']*100:.1f} | macro-F1 gap {gm['macro_f1_gap']*100:.1f} "
              f"| equalized-odds gap {gm['equalized_odds_gap']*100:.1f} points")
        for gname, v in gm["groups"].items():
            print(f"   {gname:18s} n={v['n']:5d}  acc {v['accuracy']*100:5.1f}%  macro-F1 {v['macro_f1']*100:5.1f}%")
    if baseline:
        wv = report["with_vs_without"]
        print("\n================ WITH vs WITHOUT FAIRNESS (full test set) ================")
        print(f"{'metric':30s} {'without':>9s} {'with':>9s} {'change':>9s}")
        for k, m in wv["metrics"].items():
            if k == "recall_ratio":
                print(f"{m['label']:30s} {m['without']:9.3f} {m['with']:9.3f} {m['change']:+9.3f}")
            else:
                print(f"{m['label']:30s} {m['without']*100:8.2f}% {m['with']*100:8.2f}% {m['change']*100:+8.2f}")
        print(f"{'Passes four-fifths rule':30s} {str(wv['four_fifths']['without']):>9s} {str(wv['four_fifths']['with']):>9s}")
        print("\nRecall per class (without -> with):")
        for c, v in wv["per_class_recall"].items():
            print(f"  {c:10s} (train {v['train_share']*100:4.1f}%)  {v['without']*100:5.1f}% -> {v['with']*100:5.1f}%  ({v['change']*100:+.1f})")
    print(f"\n[saved] {a.out_json} and {a.out_png} - restart the app to show these in /predict and the PDF report")


if __name__ == "__main__":
    main()
