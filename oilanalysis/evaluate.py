"""Run the paper's experiments and write their results to --out as CSV files.

    compare       every classifier, ND-LR structure and decomposition on 50 engine-grouped splits
    leakage       the same splits with row-level splitting, preprocessing fitted on all rows,
                  and settings picked on the test fold
    variability   chosen models over many engine-grouped splits
    flags         leave-one-engine-out consensus review of the expert-assigned grades
    analysts      expert terms added to each ND-LR node, with an engine bootstrap
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

from . import oil

PAPER_MODELS = ["LR", "LDA", "RF", "SVM", "DT", "XGB", "MLP",
                "LR-OVR", "LR-OVO", "SVM-OVR", "SVM-OVO"]
LEAK_MODELS = ["LR", "LDA", "RF", "SVM", "DT", "XGB", "MLP"]
REVIEW_MODELS = ["ND-LR", "LR", "RF", "XGB", "MLP"]
ARMS = {"honest": (True, False, False), "prep": (True, True, False),
        "tuned": (True, False, True), "split": (False, False, False), "all": (False, True, True)}
NODE_C = [np.inf, 10.0, 1.0, 0.1]
GRIDS = {
    "LR": [{"C": float(c), "class_weight": w} for c in np.logspace(-3, 3, 13)
           for w in [None, "balanced"]],
    "LDA": [{"solver": "svd"}] + [{"solver": "lsqr", "shrinkage": s}
                                  for s in [None, "auto", 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]],
    "RF": [{"n_estimators": n, "max_depth": d, "min_samples_leaf": l, "max_features": f}
           for n in [100, 200, 300] for d in [None, 3, 6, 10, 15]
           for l in [1, 2, 5, 10] for f in ["sqrt", None]],
    # libsvm has no iteration limit by default and a polynomial kernel at gamma=1 does not
    # converge here
    "SVM": [{"C": float(c), "gamma": gm, "kernel": k, "class_weight": w, "max_iter": 200_000}
            for c in [0.1, 1.0, 10.0, 100.0] for gm in ["scale", 0.01, 0.1, 1.0]
            for k in ["rbf", "poly"] for w in [None, "balanced"]],
    "DT": [{"max_depth": d, "min_samples_leaf": l, "criterion": cr, "class_weight": w}
           for d in [2, 3, 4, 6, 8, None] for l in [1, 2, 5, 10, 20, 50]
           for cr in ["gini", "entropy"] for w in [None, "balanced"]],
    "XGB": [{"max_depth": d, "learning_rate": lr, "n_estimators": n, "subsample": s}
            for d in [2, 3, 4, 6] for lr in [0.03, 0.1, 0.3] for n in [100, 200, 300]
            for s in [0.7, 1.0]],
    "MLP": [{"hidden_layer_sizes": h, "alpha": a, "learning_rate_init": lr, "activation": act}
            for h in [(16,), (32,), (64,), (128,), (256,), (32, 32), (64, 32), (64, 64),
                      (128, 64), (128, 128), (256, 128), (64, 64, 32)]
            for a in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
            for lr in [1e-3, 1e-2]
            for act in ["relu", "tanh"]],
}
# the honest MLP picks from oil.MLP_GRID, so the test-fold search must hold each of those settings
_HONEST_MLP = [{"learning_rate_init": 1e-3, "activation": "relu", **p} for p in oil.MLP_GRID]
GRIDS["MLP"] += [p for p in _HONEST_MLP if p not in GRIDS["MLP"]]
# a one-vs-rest or one-vs-one model is searched over the grid of the model it wraps
for _name in ["LR-OVR", "LR-OVO", "SVM-OVR", "SVM-OVO"]:
    GRIDS[_name] = [{f"estimator__{k}": v for k, v in p.items()}
                    for p in GRIDS[_name.split("-")[0]]]


def compare_split(args, tables, y, g, seed, models):
    tr, te = oil.outer_split(y, g, seed)
    Xtr_df, Xtr, Xte = oil.matrices(tables, tr, te)
    pred = oil.predict(models, Xtr_df, Xtr, y[tr], g[tr], Xte, args.classes, oil.node_rule(args))
    labels = np.array(args.classes)
    return pd.DataFrame({"seed": seed, "row": te, "grade": labels[y[te]],
                         **{m: labels[p] for m, p in pred.items()}})


def compare(args):
    d, y, g = oil.load(args)
    tables = oil.table_builder(args, d, y)
    models = list(dict.fromkeys([args.tree, args.tree + oil.SOFT,
                                 *oil.ordered_trees(args.classes), *args.models]))
    out = Parallel(n_jobs=args.jobs)(delayed(compare_split)(args, tables, y, g, s, models)
                                     for s in range(args.splits))
    pd.concat(out).to_csv(args.out / "model_comparison.csv", index=False)


def search(name, Xtr, ytr, gtr, Xte, yte):
    """Best test-fold macro-F1 over the model's grid, which contains its honest setting. A model
    with no grid is scored at its own setting only. Each MLP setting is fitted as the honest MLP
    fits its winner: epoch rule on held-out gtr groups and five initialisations averaged."""
    grid = GRIDS.get(name, [{}])
    if name == "MLP":
        preds = (oil.mlp_seed_mean(Xtr, ytr, gtr, Xte, **p).argmax(1) for p in grid)
    else:
        preds = (clone(oil.ESTIMATORS[name]).set_params(**p).fit(Xtr, ytr).predict(Xte)
                 for p in grid)
    return max((oil.scores(yte, p) for p in preds), key=lambda s: s["macroF1"])


def search_nd(args, Xtr_df, Xtr, ytr, Xte, yte):
    """Structure, node penalty and cutoffs all chosen on the test fold."""
    rule, names = oil.node_rule(args), oil.feature_order(Xtr_df)
    best = None
    for text in oil.ordered_trees(args.classes):
        tree = oil.parse_tree(text, args.classes)
        for C in NODE_C:
            node = LogisticRegression(C=C, max_iter=5000)
            p = [oil.p_left(oil.fit_node(node, split, rule(names, split), Xtr, ytr), Xte)
                 for split in tree]
            s = oil.scores(yte, oil.predict_hard(tree, p, oil.tune_cutoffs(tree, p, yte)))
            if best is None or s["macroF1"] > best["macroF1"]:
                best = s
    return best


def leakage_seed(args, tables, X_all, y, g, seed):
    rows = []
    for arm, (grouped, prep_all, tune_on_test) in ARMS.items():
        tr, te = oil.outer_split(y, g, seed, grouped)
        if prep_all:
            pp = oil.preprocessor(X_all).fit(X_all)
            Xtr_df, Xtr, Xte = X_all.iloc[tr], pp.transform(X_all.iloc[tr]), pp.transform(X_all.iloc[te])
        else:
            Xtr_df, Xtr, Xte = oil.matrices(tables, tr, te)
        if tune_on_test:
            gtr = g[tr] if grouped else np.arange(len(tr))
            res = {m: search(m, Xtr, y[tr], gtr, Xte, y[te]) for m in args.models}
            res[args.tree] = search_nd(args, Xtr_df, Xtr, y[tr], Xte, y[te])
        else:
            pred = oil.predict(args.models + [args.tree], Xtr_df, Xtr, y[tr], g[tr], Xte,
                               args.classes, oil.node_rule(args), grouped)
            res = {m: oil.scores(y[te], p) for m, p in pred.items()}
        rows += [{"seed": seed, "arm": arm, "model": m, **s} for m, s in res.items()]
    return rows


def leakage(args):
    print({m: len(GRIDS.get(m, [{}])) for m in args.models}, "settings searched on the test fold, plus",
          len(oil.ordered_trees(args.classes)) * len(NODE_C), "for ND-LR")
    d, y, g = oil.load(args)
    tables = oil.table_builder(args, d, y)
    everything = np.arange(len(y))
    X_all = tables(everything, everything)[0]
    out = Parallel(n_jobs=args.jobs)(delayed(leakage_seed)(args, tables, X_all, y, g, s)
                                     for s in range(args.splits))
    pd.DataFrame([r for rows in out for r in rows]).to_csv(args.out / "leakage_scores.csv", index=False)


def variability_split(args, tables, y, g, seed):
    tr, te = oil.outer_split(y, g, seed)
    Xtr_df, Xtr, Xte = oil.matrices(tables, tr, te)
    pred = oil.predict(args.models, Xtr_df, Xtr, y[tr], g[tr], Xte, args.classes, oil.node_rule(args))
    return [{"seed": seed, "model": m, **oil.scores(y[te], p)} for m, p in pred.items()]


def variability(args):
    d, y, g = oil.load(args)
    tables = oil.table_builder(args, d, y)
    out = Parallel(n_jobs=args.jobs)(delayed(variability_split)(args, tables, y, g, s)
                                     for s in range(args.splits))
    pd.DataFrame([r for rows in out for r in rows]).to_csv(args.out / "split_variability.csv", index=False)


def flag_engine(args, tables, y, g, tr, te):
    """Class probabilities for one held-out engine from each review model. NDs are routed
    softly, because the review needs a probability per sample. A classifier without
    predict_proba gives probability 1 to the grade it predicts."""
    Xtr_df, Xtr, Xte = oil.matrices(tables, tr, te, drop=args.without)
    rule, names = oil.node_rule(args), oil.feature_order(Xtr_df)
    ytr = y[tr]
    proba = {}
    for k in args.models:
        if k == "MLP":
            proba[k] = oil.mlp_tuned_proba(Xtr, ytr, g[tr], Xte, oil.inner_folds(Xtr, ytr, g[tr]))
        elif k in oil.ESTIMATORS:
            fit = clone(oil.ESTIMATORS[k]).fit(Xtr, ytr)
            proba[k] = (fit.predict_proba(Xte) if hasattr(fit, "predict_proba")
                        else np.eye(len(args.classes))[fit.predict(Xte)])
        else:
            tree = oil.parse_tree(args.tree if k == "ND-LR" else k, args.classes)
            nd = oil.fit_nd(oil.NODE, tree, [rule(names, s) for s in tree], Xtr, ytr)
            proba[k] = nd.predict_proba(Xte)
    return te, proba


def flags(args):
    d, y, g = oil.load(args)
    tables = oil.table_builder(args, d, y)
    out = Parallel(n_jobs=args.jobs)(delayed(flag_engine)(args, tables, y, g, tr, te)
                                     for tr, te in LeaveOneGroupOut().split(y, y, g))
    voters = list(out[0][1])
    proba = {k: np.zeros((len(y), len(args.classes))) for k in voters}
    for te, p in out:
        for k in voters:
            proba[k][te] = p[k]

    votes = np.vstack([proba[k].argmax(1) for k in voters])
    agreed = votes[0]
    flagged = (votes == agreed).all(0) & (agreed != y)
    labels = np.array(args.classes)
    res = d[[args.group, *args.keep]].copy()
    res["grade"] = labels[y]
    for k in voters:
        res[f"{k}_grade"] = labels[proba[k].argmax(1)]
    res["flagged"] = flagged
    res["consensus"] = np.where(flagged, labels[agreed], "")
    res["probability"] = np.where(
        flagged, np.mean([proba[k][np.arange(len(y)), agreed] for k in voters], axis=0), np.nan)
    res["direction"] = np.where(flagged, np.where(agreed > y, "optimistic", "cautious"), "")
    res["steps"] = np.where(flagged, np.abs(agreed - y), 0)
    res.to_csv(args.data.parent / "grade_review.csv", index=False)


def analyst_fit(X, y, rows, tree, columns, C):
    """Intercept and coefficients at each node, joined into one vector, positive toward the
    more severe (right) side. The default ridge penalty, C=1, keeps the bootstrap refits finite
    when a resample separates a node."""
    Xr, yr = X[rows], y[rows]
    out = []
    for (left, right), cols in zip(tree, columns):
        if not (np.isin(yr, left).any() and np.isin(yr, right).any()):
            return None
        fit = oil.fit_node(LogisticRegression(C=C, max_iter=5000), (left, right), cols, Xr, yr)[-1]
        out.append(np.r_[fit.intercept_[0], fit.coef_[0]])
    return np.concatenate(out)


def analyst_draw(X, y, g, tree, columns, C, seed):
    rng = np.random.default_rng(seed)
    engines = np.unique(g)
    while True:
        rows = np.concatenate([np.flatnonzero(g == m)
                               for m in rng.choice(engines, len(engines), replace=True)])
        coefs = analyst_fit(X, y, rows, tree, columns, C)
        if coefs is not None:
            return coefs


def analysts(args):
    """The ND-LR nodes refitted on the samples the chosen experts graded on the engines all of
    them graded, with an expert indicator at every node. Features are built on every sample,
    because some look back along each engine's full history."""
    d, y, g = oil.load(args)
    everything = np.arange(len(y))
    X_all = oil.table_builder(args, d, y)(everything, everything)[0].drop(columns=args.without)
    who = d[args.by].to_numpy()
    shared = set.intersection(*(set(g[who == a]) for a in args.analysts))
    keep = np.isin(who, args.analysts) & np.isin(g, list(shared))
    Xdf, y, g, who = X_all[keep], y[keep], g[keep], who[keep]
    print(f"{keep.sum()} samples from {len(shared)} engines graded by all of {args.analysts}")

    dummies = pd.get_dummies(pd.Categorical(who, categories=args.analysts), drop_first=True,
                             dtype=float)
    X = np.hstack([oil.preprocessor(Xdf).fit_transform(Xdf), dummies.to_numpy()])
    names = oil.feature_order(Xdf)
    extra = [f"{args.by}_{a}" for a in args.analysts[1:]]
    tree = oil.parse_tree(args.tree, args.classes)
    rule = oil.node_rule(args)
    columns = [rule(names, s) + list(range(len(names), len(names) + len(extra))) for s in tree]
    terms = [(i, t) for i, cols in enumerate(columns)
             for t in ["intercept", *[names[c] for c in cols[:-len(extra)]], *extra]]

    full = analyst_fit(X, y, np.arange(len(y)), tree, columns, args.C)
    draws = np.stack(Parallel(n_jobs=args.jobs)(
        delayed(analyst_draw)(X, y, g, tree, columns, args.C, s) for s in range(args.draws)))
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    nodes = [f"{oil.group_name(l, args.classes)}|{oil.group_name(r, args.classes)}" for l, r in tree]
    pd.DataFrame([{"node": nodes[i], "term": t, "coef": full[k], "lo": lo[k], "hi": hi[k],
                   "reference": args.analysts[0], "draws": args.draws}
                  for k, (i, t) in enumerate(terms)]
                 ).to_csv(args.out / "analyst_effects.csv", index=False)


def main():
    ap = oil.parser(__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare", help="every ordered ND-LR structure plus --models on grouped splits")
    p.add_argument("--models", nargs="*", default=PAPER_MODELS)
    p.add_argument("--splits", type=int, default=50)
    p = sub.add_parser("leakage", help="honest, leaked-preprocessing, test-tuned and row-level arms")
    p.add_argument("--models", nargs="+", default=LEAK_MODELS, help="run alongside the --tree ND-LR")
    p.add_argument("--splits", type=int, default=50)
    p = sub.add_parser("variability", help="--models over many grouped splits")
    p.add_argument("--models", nargs="+", default=["LR"])
    p.add_argument("--splits", type=int, default=1000)
    without = "features left out, by default the earlier-grade features prev<grade>"
    p = sub.add_parser("flags", help="unanimous leave-one-engine-out disagreements with the grade")
    p.add_argument("--models", nargs="+", default=REVIEW_MODELS,
                   help="voters: ESTIMATORS keys, MLP, ND-LR (the --tree ND-LR) or any tree")
    p.add_argument("--without", nargs="*", help=without)
    p.add_argument("--keep", nargs="*", help="columns copied into grade_review.csv next to the data")
    p = sub.add_parser("analysts", help="expert terms at each ND-LR node with an engine bootstrap")
    p.add_argument("--without", nargs="*", help=without)
    p.add_argument("--by", default="analyst")
    p.add_argument("--analysts", nargs="+", default=["R1", "R2", "R3"],
                   help="the first is the reference")
    p.add_argument("--draws", type=int, default=3000)
    p.add_argument("--C", type=float, default=1.0, help="node penalty strength, inf for none")
    args = oil.parse_args(ap)
    args.out.mkdir(parents=True, exist_ok=True)
    {"compare": compare, "leakage": leakage, "variability": variability,
     "flags": flags, "analysts": analysts}[args.command](args)


if __name__ == "__main__":
    main()
