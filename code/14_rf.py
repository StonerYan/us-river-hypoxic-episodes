"""Random forests for entry and exit, with site-blocked validation.

Purpose is attribution, not prediction: which drivers carry skill for entering
hypoxia, and which carry skill for leaving it. Folds are grouped on site so no
site appears in both training and evaluation. Importance is the loss in held-out
AUC when one predictor is permuted, so it is measured on unseen sites.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import OUT, ensure

WEATHER = ["aU_z", "aT_z", "aL_z", "U_mean", "tw_mean", "ssrd_MJ"]
NEG_PER_POS = 20      # case-control thinning for the very rare onset events
SEED = 17


def prep(df: pd.DataFrame, feats: list[str], thin: bool) -> pd.DataFrame:
    d = df.dropna(subset=feats + ["y"]).copy()
    if thin:
        rng = np.random.default_rng(SEED)
        pos = d[d["y"] == 1]
        neg = d[d["y"] == 0]
        take = min(len(neg), NEG_PER_POS * len(pos))
        neg = neg.iloc[rng.choice(len(neg), take, replace=False)]
        d = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED)
    return d


def run(d: pd.DataFrame, feats: list[str], label: str) -> dict:
    X = d[feats].to_numpy(dtype=float)
    y = d["y"].to_numpy(dtype=int)
    groups = d["site_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    aucs, aps, imps = [], [], []
    for tr, te in gkf.split(X, y, groups):
        if y[te].sum() < 10 or y[tr].sum() < 10:
            continue
        rf = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=30, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
        pi = permutation_importance(rf, X[te], y[te], scoring="roc_auc",
                                    n_repeats=8, random_state=SEED, n_jobs=-1)
        imps.append(pi.importances_mean)
    imps = np.array(imps)
    res = {
        "label": label,
        "n": int(len(d)),
        "events": int(y.sum()),
        "sites": int(pd.unique(groups).size),
        "auc_mean": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs)),
        "ap_mean": float(np.mean(aps)),
        "folds": len(aucs),
        "importance": {},
    }
    for j, f in enumerate(feats):
        v = imps[:, j]
        res["importance"][f] = {"mean": float(v.mean()), "sd": float(v.std()),
                                "lo": float(v.min()), "hi": float(v.max())}
    # one full-data forest for partial dependence
    rf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=30, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    rf.fit(X, y)
    res["pdp"] = {}
    for f in ["aU_z", "aT_z", "aL_z"]:
        if f not in feats:
            continue
        j = feats.index(f)
        pd_ = partial_dependence(rf, X, [j], grid_resolution=25,
                                 kind="average", percentiles=(0.02, 0.98))
        res["pdp"][f] = {"x": pd_["grid_values"][0].tolist(),
                         "y": pd_["average"][0].tolist()}
    return res


def main() -> int:
    ensure()
    on = pd.read_parquet(OUT / "onset_hyp2.parquet")
    rc = pd.read_parquet(OUT / "recovery_hyp2.parquet")
    rc["lspell"] = np.log(rc["spell_day"])

    out = {}
    d_on = prep(on, WEATHER, thin=True)
    out["onset"] = run(d_on, WEATHER, "entry into hypoxia")
    d_rc = prep(rc, WEATHER + ["lspell"], thin=False)
    out["recovery"] = run(d_rc, WEATHER + ["lspell"], "exit from hypoxia")
    out["recovery_weather_only"] = run(d_rc, WEATHER, "exit, weather only")
    # anomaly-only variant: absolute temperature also encodes site and season, so
    # a within-site anomaly forest is the fair nonlinear counterpart to the logits
    ANOM = ["aU_z", "aT_z", "aL_z"]
    out["onset_anom"] = run(prep(on, ANOM, thin=True), ANOM, "entry, anomalies only")
    out["recovery_anom"] = run(prep(rc, ANOM, thin=False), ANOM, "exit, anomalies only")
    out["notes"] = {
        "onset_thinning": f"all events plus {NEG_PER_POS} matched non-events per event",
        "validation": "5-fold GroupKFold on site; permutation importance on held-out folds",
    }
    (OUT / "rf.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for k in ["onset", "recovery", "recovery_weather_only", "onset_anom", "recovery_anom"]:
        r = out[k]
        print(f"\n[{k}] n={r['n']} ev={r['events']} sites={r['sites']} "
              f"AUC={r['auc_mean']:.3f}+-{r['auc_sd']:.3f} AP={r['ap_mean']:.3f}")
        rank = sorted(r["importance"].items(), key=lambda kv: -kv[1]["mean"])
        for f, v in rank:
            print(f"   {f:>10s}  dAUC={v['mean']:+.4f} +- {v['sd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
