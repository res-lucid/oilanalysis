"""Data loading, the paper's features, nested dichotomies, classifiers and splits."""
import argparse
import copy
import re
import warnings
from itertools import product
from pathlib import Path

import ndscape as nds
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import (GroupShuffleSplit, ShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# sklearn 1.8 warns about C=np.inf, which is its own documented way to fit without a penalty
warnings.filterwarnings("ignore", message="Setting penalty=None")

HERE = Path(__file__).resolve().parent
MEASURED = ["Fe", "Cu", "Pb", "Al", "Si", "K", "NIT", "TBN", "ST", "V40", "oilhours", "meterread"]
VALID = {"oilhours": (0, 2000), "meterread": (0, 150000)}
RANKED = ["oil_life"]

METRICS = {"acc": accuracy_score, "balAcc": balanced_accuracy_score,
           "macroF1": lambda a, b: f1_score(a, b, average="macro"), "MCC": matthews_corrcoef}

NODE = LogisticRegression(C=np.inf, max_iter=5000)
ESTIMATORS = {
    "LR": LogisticRegression(max_iter=2000),
    "LDA": LinearDiscriminantAnalysis(),
    "RF": RandomForestClassifier(n_estimators=200, random_state=0),
    "SVM": SVC(random_state=0),
    "DT": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0),
    "XGB": XGBClassifier(n_jobs=1, tree_method="hist", eval_metric="mlogloss", verbosity=0,
                         random_state=0),
    "LR-OVR": OneVsRestClassifier(LogisticRegression(max_iter=2000)),
    "LR-OVO": OneVsOneClassifier(LogisticRegression(max_iter=2000)),
    "SVM-OVR": OneVsRestClassifier(SVC(random_state=0)),
    "SVM-OVO": OneVsOneClassifier(SVC(random_state=0)),
}
MLP_GRID = [{"hidden_layer_sizes": h, "alpha": a}
            for h in ((32,), (64, 32), (128, 64), (256, 128, 64)) for a in (1e-4, 1e-2)]
MLP_SEEDS = range(5)
SOFT = " soft"


def parser(description):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data/Multiclass_oil_analysis_on_diesel_trucks.csv")
    ap.add_argument("--target", default="grade")
    ap.add_argument("--group", default="engine", help="column identifying the engine")
    ap.add_argument("--classes", nargs="+", default=list("ABCX"), help="grades, healthiest first")
    ap.add_argument("--tree", default="(A(B(CX)))", help="the nested dichotomy reported as ND-LR")
    ap.add_argument("--features", nargs="+",
                    help="use these columns as they are instead of building the paper's features")
    ap.add_argument("--only", nargs="*", metavar="FEATURE:NODE",
                    help="offer a feature only at the node over these grades")
    ap.add_argument("--drop", nargs="*", metavar="FEATURE:NODE",
                    help="withhold a feature at the node over these grades")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--jobs", type=int, default=4)
    return ap


def parse_args(ap):
    """Options left unset take the paper's values when the paper's features are built, and are
    empty when --features is given."""
    args = ap.parse_args()
    paper = {"only": ["V40_low:CX", "soot_z:CX"], "drop": ["is_truck:CX"],
             "without": [f"prev{c}" for c in args.classes],
             "keep": ["sample", "date", "analyst", "note"]}
    for name, value in paper.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, value if args.features is None else [])
    return args


def load(args):
    """Samples, integer grades and engine ids, without rows whose grade is not in --classes.
    The paper's features need rows in engine and date order, because some look back along
    the engine's history. The sort is stable so same-day samples keep their file order."""
    d = pd.read_csv(args.data, low_memory=False, dtype={args.target: str})
    d = d[d[args.target].isin(args.classes)]
    if args.features is None:
        d = d.sort_values([args.group, "date"], kind="stable")
    d = d.reset_index(drop=True)
    y = d[args.target].map({c: i for i, c in enumerate(args.classes)}).to_numpy()
    return d, y, d[args.group].to_numpy()


def table_builder(args, d, y):
    """Train and test feature tables for a split. Fitted feature statistics use training rows
    only."""
    if args.features:
        X = below_detection_zero(d[args.features]).apply(pd.to_numeric, errors="coerce")
        return lambda tr, te: (X.iloc[tr], X.iloc[te])

    def tables(tr, te):
        X = oil_features(d, y, args.group, args.classes, tr)
        return X.iloc[tr], X.iloc[te]
    return tables


def below_detection_zero(d):
    """`<1` is read as 0 and any other text becomes missing when the columns are read as numbers.
    As missing, `<1` would be median-imputed above the readings recorded as an exact 0."""
    return d.replace("<1", "0")


def oil_features(d, y, group, classes, fit_rows):
    v = {c: pd.to_numeric(x, errors="coerce").to_numpy()
         for c, x in below_detection_zero(d[MEASURED]).items()}
    fit = np.zeros(len(d), bool)
    fit[fit_rows] = True
    engine = d[group].to_numpy()

    oilhours, meterread = (np.where((v[c] >= lo) & (v[c] <= hi), v[c], np.nan)
                           for c, (lo, hi) in VALID.items())
    interval = d["service_interval"].to_numpy(float)

    Si, K = v["Si"], v["K"]
    si_on = Si > np.nanpercentile(Si[fit & (y == 0)], 90)
    k_on = K > np.nanpercentile(K[fit & (y == 0)], 90)
    si_level = np.where(si_on, np.log1p(Si) - np.log1p(Si[fit & si_on]).mean(), 0.0)
    k_level = np.where(k_on, np.log1p(K) - np.log1p(K[fit & k_on]).mean(), 0.0)

    st = pd.Series(v["ST"])
    by = st.groupby(engine)
    st_mean = by.transform(lambda s: s.shift(1).expanding().mean())
    st_sd = by.transform(lambda s: s.shift(1).expanding().std())
    soot_z = ((st - st_mean) / st_sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    model = d["model"].to_numpy()
    ref = pd.Series(model).map(pd.Series(v["V40"][fit]).groupby(model[fit]).median()).to_numpy(float)
    ref = np.where(np.isnan(ref), np.nanmedian(v["V40"][fit]), ref)

    prev = pd.Series(y).groupby(engine).shift(1)
    return pd.DataFrame({
        "Fe": v["Fe"], "Cu": v["Cu"], "Al": v["Al"], "Pb": v["Pb"],
        "Si_elevated": si_on.astype(int), "Si_level": si_level,
        "K_elevated": k_on.astype(int), "K_level": k_level,
        "coolant_combined": np.where(si_on, np.log1p(Si), 0.0) * np.where(k_on, np.log1p(K), 0.0),
        "NIT": v["NIT"], "TBN": v["TBN"],
        "oil_life": oilhours / interval,
        "meterread": meterread,
        "prev_oilchanged": d.groupby(group)["oilchanged"].shift(1).eq(1).astype(float).to_numpy(),
        **{f"prev{c}": (prev == i).astype(int).to_numpy() for i, c in enumerate(classes)},
        "is_truck": d["truck"].to_numpy(int),
        "soot_z": soot_z.to_numpy(),
        "V40_low": np.clip(ref - v["V40"], 0, None),
    })


def column_groups(X):
    binary = [c for c in X if X[c].dropna().isin([0, 1]).all()]
    ranked = [c for c in RANKED if c in X]
    return [c for c in X if c not in binary + ranked], ranked, binary


def feature_order(X):
    cont, ranked, binary = column_groups(X)
    return cont + ranked + binary


def preprocessor(X):
    cont, ranked, binary = column_groups(X)
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cont),
        ("rank", make_pipeline(SimpleImputer(strategy="median"),
                               QuantileTransformer(output_distribution="normal", n_quantiles=500,
                                                   random_state=0)), ranked),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ])


def matrices(tables, tr, te, drop=()):
    Xtr_df, Xte_df = tables(tr, te)
    Xtr_df, Xte_df = Xtr_df.drop(columns=list(drop)), Xte_df.drop(columns=list(drop))
    pp = preprocessor(Xtr_df).fit(Xtr_df)
    return Xtr_df, pp.transform(Xtr_df), pp.transform(Xte_df)


def separator(classes):
    """Single-letter grades are written run together, as in (A(B(CX))) and CX. Longer names are
    separated by spaces in a tree, as in (Normal (Caution Critical)), and by + in a node name."""
    return "" if all(len(c) == 1 for c in classes) else " "


def grade_ids(text, classes):
    return tuple(classes.index(c) for c in (text.split("+") if separator(classes) else text))


def group_name(side, classes):
    return ("+" if separator(classes) else "").join(classes[i] for i in side)


def node_rule(args):
    """Column positions each ND-LR node may use, from --only and --drop."""
    def parse(items):
        return {f: frozenset(grade_ids(n, args.classes)) for f, n in (i.split(":") for i in items)}
    only, drop = parse(args.only), parse(args.drop)

    def columns(names, split):
        node = frozenset(split[0] + split[1])
        return [i for i, c in enumerate(names) if only.get(c, node) == node and drop.get(c) != node]
    return columns


def parse_tree(text, classes):
    """'(A(B(CX)))' -> its splits in preorder, each a (left class ids, right class ids) pair."""
    tokens = iter(re.findall(r"[()]|[^\s()]+" if separator(classes) else r"\S", text))

    def read():
        t = next(tokens)
        if t != "(":
            return grade_ids(t, classes), []
        left, left_splits = read()
        right, right_splits = read()
        next(tokens)
        return left + right, [(left, right)] + left_splits + right_splits
    return read()[1]


def ordered_trees(classes):
    """Every nested dichotomy whose splits keep the grades in order."""
    sep = separator(classes)

    def trees(labels):
        if len(labels) == 1:
            return labels
        return [f"({a}{sep}{b})" for k in range(1, len(labels))
                for a in trees(labels[:k]) for b in trees(labels[k:])]
    return trees(list(classes))


def fit_node(est, split, cols, X, y):
    """ndscape's fit of one split, with est seeing only the columns `cols`. As in ndscape,
    class 0 is the left side and class 1 the right."""
    base = make_pipeline(ColumnTransformer([("node", "passthrough", cols)]), est)
    return nds.ND([split], split[0] + split[1]).fit(X, y, base=base).models[split]


def fit_nd(est, tree, columns, X, y):
    """ndscape ND of the tree, each node fitted by fit_node on its own columns."""
    models = [fit_node(est, s, cols, X, y) for s, cols in zip(tree, columns)]
    return nds.ND.from_trained(tree, sorted(tree[0][0] + tree[0][1]), models)


def fit_tuned(X, y, groups, tree, columns=None, est=NODE):
    """ND-LR with each node's cutoff tuned on grouped inner folds of the training rows. y holds
    grade positions, healthiest first, and columns defaults to every column at every node.
    Returns the ndscape ND, whose predict routes softly, and the cutoffs for predict_tuned."""
    columns = columns or [list(range(X.shape[1]))] * len(tree)
    inner = inner_folds(X, y, groups)
    oof, models = zip(*(node_probs(est, s, cols, X, y, inner) for s, cols in zip(tree, columns)))
    nd = nds.ND.from_trained(tree, sorted(tree[0][0] + tree[0][1]), list(models))
    return nd, tune_cutoffs(tree, list(oof), y)


def predict_tuned(nd, cutoffs, X):
    return predict_hard(nd.tree, [p_left(nd.models[s], X) for s in nd.tree], cutoffs)


def p_left(model, X):
    return model.predict_proba(X)[:, 0]


def node_probs(est, split, cols, Xtr, ytr, inner):
    """P(left) out-of-fold on the training rows, and the node refitted on all of them."""
    oof = np.zeros(len(ytr))
    for fit_idx, val_idx in inner:
        oof[val_idx] = p_left(fit_node(est, split, cols, Xtr[fit_idx], ytr[fit_idx]), Xtr[val_idx])
    return oof, fit_node(est, split, cols, Xtr, ytr)


def predict_hard(tree, p_left, cutoffs):
    branch = {frozenset(l + r): (l, r, p, c) for (l, r), p, c in zip(tree, p_left, cutoffs)}
    pred = np.empty(len(p_left[0]), int)

    def descend(node, mask):
        if len(node) == 1:
            pred[mask] = next(iter(node))
            return
        left, right, p, cut = branch[node]
        descend(frozenset(left), mask & (p >= cut))
        descend(frozenset(right), mask & (p < cut))

    descend(frozenset(tree[0][0] + tree[0][1]), np.ones(len(pred), bool))
    return pred


def tune_cutoffs(tree, p_left, y):
    """Cutoffs maximising macro-F1, on a 0.1 grid. A finer grid overfits the cutoff surface."""
    best, cuts = -np.inf, None
    for t in product(np.round(np.arange(0.02, 0.99, 0.1), 4), repeat=len(tree)):
        s = f1_score(y, predict_hard(tree, p_left, t), average="macro")
        if s > best:
            best, cuts = s, np.array(t)
    return cuts


def balanced_rows(y, seed=0):
    rng = np.random.default_rng(seed)
    per_class = [np.flatnonzero(y == c) for c in np.unique(y)]
    return np.concatenate([rng.choice(c, max(map(len, per_class))) for c in per_class])


def fit_mlp(X, y, groups, seed=0, max_epochs=300, patience=30, **params):
    """One net whose epoch count is chosen on a fifth of the engines held out before the
    class-balanced oversample is drawn. MLPClassifier's own early stopping splits after
    oversampling, so copies of one row land on both sides of its validation cut."""
    fit, val = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, groups))
    rows = balanced_rows(y[fit], seed)
    Xf, yf = X[fit][rows], y[fit][rows]
    present = np.unique(y[val])
    rng = np.random.default_rng(seed)
    # params last, so a grid that sets the step size overrides the default
    net = MLPClassifier(**{"batch_size": 128, "learning_rate_init": 1e-3, "random_state": seed,
                           **params})
    best, kept, bad = -np.inf, None, 0
    for _ in range(max_epochs):
        order = rng.permutation(len(yf))
        net.partial_fit(Xf[order], yf[order], classes=np.unique(y))
        s = f1_score(y[val], net.predict(X[val]), average="macro", labels=present)
        if s > best:
            best, kept, bad = s, copy.deepcopy(net), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return kept


def mlp_seed_mean(Xtr, ytr, gtr, Xte, **params):
    """Class probabilities averaged over five initialisations of one setting."""
    return np.mean([fit_mlp(Xtr, ytr, gtr, seed=s, **params).predict_proba(Xte)
                    for s in MLP_SEEDS], axis=0)


def mlp_tuned_proba(Xtr, ytr, gtr, Xte, inner):
    """Architecture and penalty chosen on the inner folds, then five initialisations averaged."""
    scores = [np.mean([f1_score(ytr[v], fit_mlp(Xtr[f], ytr[f], gtr[f], **p).predict(Xtr[v]),
                                average="macro") for f, v in inner]) for p in MLP_GRID]
    return mlp_seed_mean(Xtr, ytr, gtr, Xte, **MLP_GRID[int(np.argmax(scores))])


def outer_split(y, groups, seed, grouped=True):
    Z = np.zeros((len(y), 1))
    if grouped:
        return next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(Z, y, groups))
    return next(ShuffleSplit(1, test_size=0.2, random_state=seed).split(Z))


def inner_folds(Xtr, ytr, gtr, grouped=True):
    if grouped:
        return list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xtr, ytr, gtr))
    return list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr))


def predict(models, Xtr_df, Xtr, ytr, gtr, Xte, classes, rule, grouped=True):
    """Test predictions for each named model: a key of ESTIMATORS, MLP, a tree such as
    '(A(B(CX)))', or a tree plus SOFT (ndscape's product-of-probabilities routing, no cutoffs).
    Trees share node fits for shared splits. Without grouping, the MLP holds out rows, not
    engines, for its epoch count."""
    names = feature_order(Xtr_df)
    inner = inner_folds(Xtr, ytr, gtr, grouped)
    out, nodes = {}, {}
    for m in models:
        if m in ESTIMATORS:
            out[m] = clone(ESTIMATORS[m]).fit(Xtr, ytr).predict(Xte)
        elif m == "MLP":
            groups = gtr if grouped else np.arange(len(ytr))
            out[m] = mlp_tuned_proba(Xtr, ytr, groups, Xte, inner).argmax(1)
        else:
            tree = parse_tree(m.removesuffix(SOFT), classes)
            for s in tree:
                if s not in nodes:
                    nodes[s] = node_probs(NODE, s, rule(names, s), Xtr, ytr, inner)
            if m.endswith(SOFT):
                nd = nds.ND.from_trained(tree, range(len(classes)), [nodes[s][1] for s in tree])
                out[m] = nd.predict(Xte)
            else:
                cuts = tune_cutoffs(tree, [nodes[s][0] for s in tree], ytr)
                out[m] = predict_hard(tree, [p_left(nodes[s][1], Xte) for s in tree], cuts)
    return out


def scores(y, pred):
    return {k: f(y, pred) for k, f in METRICS.items()}
