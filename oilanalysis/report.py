"""Figures, LaTeX tables and the numbers quoted in the paper, written to --out/figures and --out.

    data          sample counts, variable screening, correlation and feature figures (oil data only)
    model         full-data ND-LR coefficients, and node ROC curves and cutoffs on the first split
    compare       results table, MCC boxes, paired win shares and differences, per-grade recall
                  and precision, ND-LR without tuned cutoffs, within-one-grade rate,
                  ND-LR structures and decompositions (needs model_comparison.csv)
    leakage       leaked-minus-honest gaps and leaked scores (needs leakage_scores.csv)
    variability   percentile spans and histograms (needs split_variability.csv)
    flags         flag counts, expert directions and disagreement tables (needs grade_review.csv)
    analysts      when each expert graded, and the expert coefficient table
                  (needs analyst_effects.csv)
    all           every one of the above
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

from . import oil

FULL, WIDE = 6.47, 5.5
FOCUS, FOCUS_LINE = "#56B4E9", "#0072B2"
REFERENCE, REFERENCE_LINE = "#B8860B", "#7A5A07"
THIRD, FOURTH = "#1B7837", "#CC79A7"
DARK, NEUTRAL, PALE = "#333333", "#BBBBBB", "#DDDDDD"
GRADES = LinearSegmentedColormap.from_list("grades", ["#117733", "#DDCC77", "#CC6677", "#882255"])
FLAT = ["LR", "LDA", "RF", "SVM", "DT", "XGB", "MLP"]
NUMBER_WORDS = dict(enumerate("zero one two three four five six seven eight nine ten".split()))
SHORT_NAMES = {"acc": "Accuracy", "balAcc": "Balanced accuracy", "macroF1": "Macro-F1", "MCC": "MCC"}
METRIC_NAMES = {"acc": "Accuracy", "balAcc": "Balanced accuracy", "macroF1": "Macro-F1",
                "MCC": "Matthews correlation coefficient"}
VARIABLE_GROUPS = [("Wear and debris", ["Fe", "Cu", "Pb", "Al", "PQI"], FOCUS_LINE, "-"),
                   ("Contamination", ["Si", "Na", "K", "Mo", "B", "ST"], THIRD, "--"),
                   ("Oil condition", ["OXI", "NIT", "SUL", "V40", "TBN"], REFERENCE, "-."),
                   ("Additives", ["Ca", "Zn", "P", "Mg"], DARK, ":")]
TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
       "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def style():
    for f in (oil.HERE / "fonts").glob("*.otf"):
        font_manager.fontManager.addfont(f)
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.unicode_minus": False,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "grid.linewidth": 0.4,
        "grid.alpha": 0.4, "grid.color": NEUTRAL, "hatch.linewidth": 0.6,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save(args, fig, name):
    folder = args.out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}")
    plt.close(fig)


def gridlines(ax, axis="x"):
    ax.grid(axis=axis)
    ax.set_axisbelow(True)


def boxes(ax, values, names):
    bp = ax.boxplot(values, orientation="horizontal", patch_artist=True, tick_labels=names,
                    widths=0.55, medianprops=dict(color=DARK, lw=0.9),
                    whiskerprops=dict(color=DARK, lw=0.7), capprops=dict(color=DARK, lw=0.7),
                    flierprops=dict(marker=".", ms=3, mfc=NEUTRAL, mec="none"))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=FOCUS if i == 0 else NEUTRAL, edgecolor=DARK, linewidth=0.5)
    if ax.get_ylim()[0] < ax.get_ylim()[1]:
        ax.invert_yaxis()
    gridlines(ax)


def with_nd(args, frame):
    return frame.replace({"model": {args.tree: "ND-LR", args.tree + oil.SOFT: "ND-LR" + oil.SOFT}})


def mean_ci(d):
    """Mean of paired differences and a 95% t interval on that mean."""
    d = np.asarray(d)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - half, d.mean() + half


def data(args):
    d, y, _ = oil.load(args)
    print(f"{len(d)} samples, {d[args.group].nunique()} engines, {d['date'].min()} to {d['date'].max()}")
    print("grades:", d[args.target].value_counts().reindex(args.classes).to_dict())
    print("expert recorded:", int((d["analyst"] != "Unknown").sum()))
    bad = {c: ~pd.to_numeric(d[c], errors="coerce").between(*r) for c, r in oil.VALID.items()}
    invalid = bad["oilhours"] | bad["meterread"]
    hours = pd.to_numeric(d["oilhours"], errors="coerce")[invalid]
    print(f"invalid oilhours or meterread: {int(invalid.sum())} "
          f"(oilhours {int(bad['oilhours'].sum())}, meterread {int(bad['meterread'].sum())}), "
          f"oilhours range {hours.min():.0f} to {hours.max():.0f}")

    variables = [v for _, vs, _, _ in VARIABLE_GROUPS for v in vs] + list(oil.VALID)
    for v in ["Fe", "Cu", "Al", "Pb", "Si", "K", "B", "Mo", "TBN"]:
        print(f"{v}: {pd.to_numeric(d[v], errors='coerce').isna().mean():.1%} not numeric "
              f"({(d[v] == '<1').sum()} recorded as <1, {d[v].isna().sum()} empty)")
    raw = oil.below_detection_zero(d[variables]).apply(pd.to_numeric, errors="coerce")
    for c, (lo, hi) in oil.VALID.items():
        raw[c] = raw[c].where(raw[c].between(lo, hi))
    corr = raw.corr()
    for a, b in [("Na", "K"), ("Na", "Mo"), ("K", "Mo"), ("OXI", "NIT"), ("SUL", "NIT"),
                 ("OXI", "SUL"), ("B", "K"), ("Si", "K"), ("Ca", "Zn"), ("Ca", "P"), ("Zn", "P")]:
        print(f"r({a}, {b}) = {corr.loc[a, b]:.3f}")
    print(f"PQI: {(raw['PQI'] == 0).mean():.1%} zero, {raw['PQI'].nunique()} distinct values")

    fig, ax = plt.subplots(figsize=(FULL, 7.0))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    lum = im.cmap(im.norm(corr.to_numpy()))[..., :3] @ [0.299, 0.587, 0.114]
    for i in range(len(variables)):
        for j in range(len(variables)):
            ax.text(j, i, f"{corr.iat[i, j]:.1f}".lstrip("0").replace("-0.", "-."), ha="center",
                    va="center", fontsize=7, color="white" if lum[i, j] < 0.45 else DARK)
    ax.set_xticks(range(len(variables)), variables, rotation=90, fontsize=8)
    ax.set_yticks(range(len(variables)), variables, fontsize=8)
    ax.tick_params(length=0)
    start = 0
    for name, vs, colour, dash in VARIABLE_GROUPS:
        ax.add_patch(plt.Rectangle((start - .5, start - .5), len(vs), len(vs), fill=False,
                                   edgecolor=colour, ls=dash, lw=1.3, label=name, clip_on=False))
        start += len(vs)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson correlation")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    save(args, fig, "modelLR_feature_correlation")

    everything = np.arange(len(y))
    X = oil.table_builder(args, d, y)(everything, everything)[0]
    ranked = oil.preprocessor(X).fit_transform(X)[:, oil.feature_order(X).index("oil_life")]
    binary = oil.column_groups(X)[2]
    fired = {"Si_level": "detected", "K_level": "detected", "coolant_combined": "both fired"}
    colours = GRADES(np.linspace(0, 1, len(args.classes)))
    # fewest empty cells, so no part-filled last row, and 4 wide on a tie
    n = X.shape[1]
    ncol = min((4, 3), key=lambda k: -n % k)
    rows = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(rows, ncol, figsize=(FULL, (0.82 if ncol == 3 else 1.45) * rows))
    for ax, feat in zip(axes.ravel(), X.columns):
        v = X[feat].to_numpy(float)
        if feat in binary:
            rate = [v[y == c].mean() for c in range(len(args.classes))]
            ax.bar(range(len(rate)), rate, color=colours, width=0.65, alpha=0.55)
            for c, r in enumerate(rate):
                ax.text(c, r + 0.05 * max(rate), f"{r:.0%}", ha="center", fontsize=7)
            ax.set_xticks(range(len(rate)), args.classes)
            ax.set_ylim(0, 1.35 * max(rate))
            ax.set_title(feat, fontsize=7)
            continue
        yy, title = y, feat
        if feat in fired:
            keep = v != 0
            v, yy = v[keep], y[keep]
            title = f"{feat}, n={keep.sum()}"
            if feat == "coolant_combined":
                v = np.log1p(v.clip(0))
        elif feat == "oil_life":
            v = ranked
        elif feat in ("Fe", "Cu"):
            v = np.log1p(np.clip(v, 0, None))
        lo, hi = np.nanpercentile(v, [0.5, 99.5])
        fin = v[np.isfinite(v)]
        # each grade scaled to unit area so the small X class is as visible as B. A smooth curve
        # would blur a few-valued feature or a large point mass (V40_low at 0, Si_level at its
        # detection floor), so those stay histograms, one bin per integer where they are integers
        spike = pd.Series(fin).value_counts(normalize=True).iloc[0] >= 0.3
        if len(np.unique(fin)) <= 20 or spike:
            if np.all(fin == np.round(fin)) and hi - lo <= 30:
                bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5)
            else:
                bins = np.linspace(lo, hi, 26)
            for c in range(len(args.classes)):
                vc = v[yy == c]
                ax.hist(vc, bins=bins, density=True, histtype="stepfilled", alpha=0.15,
                        color=colours[c])
                ax.hist(vc, bins=bins, density=True, histtype="step", lw=0.9, color=colours[c])
            ax.set_xlim(bins[0], bins[-1])
        else:
            grid = np.linspace(lo, hi, 300)
            for c in range(len(args.classes)):
                dens = stats.gaussian_kde(v[(yy == c) & np.isfinite(v)])(grid)
                ax.fill_between(grid, dens, alpha=0.15, color=colours[c], lw=0)
                ax.plot(grid, dens, lw=0.9, color=colours[c])
            ax.set_xlim(lo, hi)
            ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.set_title(title, fontsize=7)
    for ax in axes.ravel():
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=colours[c], ec=colours[c], alpha=0.55,
                                      label=k) for c, k in enumerate(args.classes)],
               title="expert-assigned grade", loc="lower center", ncol=len(args.classes),
               bbox_to_anchor=(0.5, 1.0), fontsize=7, title_fontsize=7)
    save(args, fig, "modelLR_feature_dists")


def model(args):
    d, y, g = oil.load(args)
    tables = oil.table_builder(args, d, y)
    rule = oil.node_rule(args)
    tree = oil.parse_tree(args.tree, args.classes)
    everything = np.arange(len(y))
    X = tables(everything, everything)[0]
    Xall, names = oil.preprocessor(X).fit_transform(X), oil.feature_order(X)

    coefs = {}
    for left, right in tree:
        cols = rule(names, (left, right))
        fit = oil.fit_node(oil.NODE, (left, right), cols, Xall, y)[-1]
        coefs[(left, right)] = pd.Series(fit.coef_[0], index=[names[i] for i in cols])
    table = pd.DataFrame({f"{oil.group_name(l, args.classes)}|{oil.group_name(r, args.classes)}": c
                          for (l, r), c in coefs.items()}).reindex(names)
    print("full-data ND-LR node coefficients, positive toward the more severe side")
    print(table.round(2).to_string())
    earlier = [f"prev{c}" for c in args.classes]
    for node in table if set(earlier) <= set(table.index) else []:
        p = table.loc[earlier, node]
        print(f"{node}: prev{args.classes[-1]} minus the largest other earlier grade "
              f"{p.iloc[-1] - p.iloc[:-1].max():+.2f}, largest is {p.idxmax()}")

    fig, axes = plt.subplots(1, len(tree), figsize=(FULL, 3.6))
    for ax, (left, right) in zip(np.atleast_1d(axes), tree):
        s = coefs[(left, right)].sort_values()
        for i, v in enumerate(s):
            ax.barh(i, v, color=REFERENCE if v < 0 else FOCUS, hatch="///" if v < 0 else "",
                    edgecolor=DARK, lw=0.4)
        ax.set_yticks(range(len(s)), s.index)
        ax.axvline(0, color=DARK, lw=0.6)
        ax.set_xlabel("coefficient")
        ax.set_title(f"{oil.group_name(left, args.classes)} vs {oil.group_name(right, args.classes)}")
        gridlines(ax)
    fig.tight_layout()
    save(args, fig, "modelLR_coefs_" + "-".join(oil.group_name(l, args.classes) for l, _ in tree))

    tr, te = oil.outer_split(y, g, 0)
    Xtr_df, Xtr, _ = oil.matrices(tables, tr, te)
    inner = oil.inner_folds(Xtr, y[tr], g[tr])
    tr_names = oil.feature_order(Xtr_df)
    oof = [oil.node_probs(oil.NODE, s, rule(tr_names, s), Xtr, y[tr], inner)[0] for s in tree]
    cuts = oil.tune_cutoffs(tree, oof, y[tr])
    fig, axes = plt.subplots(1, len(tree), figsize=(FULL, 2.7))
    for k, (ax, (left, right)) in enumerate(zip(np.atleast_1d(axes), tree)):
        node = np.isin(y[tr], left + right)
        is_left, p = np.isin(y[tr][node], left).astype(int), oof[k][node]
        fpr, tpr, _ = roc_curve(is_left, p)
        auc = roc_auc_score(is_left, p)
        print(f"{oil.group_name(left, args.classes)} vs {oil.group_name(right, args.classes)}: "
              f"out-of-fold AUC {auc:.3f}, cutoff {cuts[k]:.2f}")
        ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.7)
        ax.plot(fpr, tpr, color="black", lw=1.2)
        ax.plot((p[is_left == 0] >= cuts[k]).mean(), (p[is_left == 1] >= cuts[k]).mean(),
                "o", ms=4.5, mfc="white", mec="black", mew=1.0)
        ax.text(0.97, 0.06, f"AUC {auc:.2f}\ncut {cuts[k]:.2f}", ha="right", va="bottom", fontsize=7)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="false positive rate",
               title=f"{oil.group_name(left, args.classes)} vs {oil.group_name(right, args.classes)}")
        for spine in ax.spines.values():
            spine.set_visible(True)
    np.atleast_1d(axes)[0].set_ylabel("true positive rate")
    fig.tight_layout()
    save(args, fig, "modelLR_nd_roc")


def split_scores(args, preds):
    models = [c for c in preds if c not in ("seed", "row", "grade")]
    rows = [{"seed": s, "model": m, **oil.scores(part["grade"], part[m])}
            for s, part in preds.groupby("seed") for m in models]
    return with_nd(args, pd.DataFrame(rows))


def results_table(mean, splits):
    best = mean.max()
    body = [f"{name} & " + " & ".join(f"\\textbf{{{v:.3f}}}" if v == best[m] else f"{v:.3f}"
                                      for m, v in row.items()) + r" \\"
            for name, row in mean.iterrows()]
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\small",
                      rf"\caption{{Mean of each metric over the {splits} grouped evaluation splits.}}",
                      r"\label{tab:model_results}", r"\begin{tabular}{lcccc}", r"\hline",
                      r"Model & Accuracy & Balanced acc. & Macro-F1 & MCC \\", r"\hline",
                      *body, r"\hline", r"\end{tabular}", r"\end{table}"])


def per_grade(args, preds, models):
    rows = []
    for s, part in preds.groupby("seed"):
        for m in models:
            kw = dict(labels=args.classes, average=None, zero_division=0)
            r = recall_score(part["grade"], part[m], **kw)
            p = precision_score(part["grade"], part[m], **kw)
            rows += [{"seed": s, "model": m, "grade": c, "recall": r[i], "precision": p[i]}
                     for i, c in enumerate(args.classes)]
    return pd.DataFrame(rows)


def grades_table(mean, nd, base, splits):
    def pair(metric, grade):
        a, b = mean.loc[grade, (metric, base)], mean.loc[grade, (metric, nd)]
        return [f"\\textbf{{{v:.3f}}}" if v == max(a, b) else f"{v:.3f}" for v in (a, b)]

    body = [f"${g}$ & " + " & ".join(pair("recall", g) + pair("precision", g)) + r" \\"
            for g in mean.index]
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\small",
                      rf"\caption{{Mean recall and precision for each grade over the {splits} "
                      r"engine-grouped splits.}",
                      r"\label{tab:class_recall}", r"\begin{tabular}{lcccc}", r"\hline",
                      r" & \multicolumn{2}{c}{Recall} & \multicolumn{2}{c}{Precision} \\",
                      f"Grade & {base} & ND-LR & {base} & ND-LR \\\\", r"\hline",
                      *body, r"\hline", r"\end{tabular}", r"\end{table}"])


def compare(args):
    preds = pd.read_csv(args.out / "model_comparison.csv")
    per_split = split_scores(args, preds)
    mean = per_split.groupby("model")[list(oil.METRICS)].mean()
    splits = per_split["seed"].nunique()
    flat = [m for m in FLAT if m in mean.index]
    args.table_models, args.comparators = args.table_models or flat, args.comparators or flat
    table = mean.loc[["ND-LR", *args.table_models]]
    print(table.round(3).to_string())
    (args.out / "table_results.tex").write_text(results_table(table, splits) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(WIDE, 2.7))
    boxes(ax, [per_split.loc[per_split["model"] == m, "MCC"] for m in table.index], table.index)
    ax.set_xlabel("Matthews correlation coefficient")
    fig.tight_layout()
    save(args, fig, "modelLR_nd_vs_flat_mcc")

    wide = per_split.pivot(index="seed", columns="model")
    counts = {}
    for m in oil.METRICS:
        for base in args.comparators:
            diff = wide[(m, "ND-LR")] - wide[(m, base)]
            counts[(m, base)] = {"ND-LR higher": (diff > 0).mean(), "comparator higher": (diff < 0).mean(),
                                 "tie": (diff == 0).mean()}
    counts = pd.DataFrame(counts).T
    print("\nshare of splits on which ND-LR scored higher")
    print(counts["ND-LR higher"].unstack(0)[list(oil.METRICS)].loc[args.comparators].round(2).to_string())
    base = args.comparators[0]
    for nd in ["ND-LR", "ND-LR" + oil.SOFT]:
        print(f"\n{nd} minus {base}, mean over paired splits with 95% t interval")
        for m in oil.METRICS:
            mu, lo, hi = mean_ci(wide[(m, nd)] - wide[(m, base)])
            print(f"{m:8s} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    grade = with_nd(args, per_grade(args, preds, [args.tree, base]))
    mean_grade = (grade.groupby(["model", "grade"])[["recall", "precision"]].mean().unstack(0)
                  .reindex(args.classes))
    print(f"\nmean recall and precision per grade, ND-LR and {base}")
    print(mean_grade.round(3).to_string())
    print(f"ND-LR minus {base} recall, mean over paired splits with 95% t interval")
    by_seed = grade.pivot_table(index=["grade", "seed"], columns="model", values="recall")
    for c in args.classes:
        mu, lo, hi = mean_ci(by_seed.loc[c, "ND-LR"] - by_seed.loc[c, base])
        print(f"{c} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (args.out / "table_grades.tex").write_text(
        grades_table(mean_grade, "ND-LR", base, splits) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.0), sharey=True, sharex=True)
    yp = np.arange(len(args.comparators))
    for ax, m in zip(axes.ravel(), oil.METRICS):
        c = counts.loc[m].loc[args.comparators] * 100
        ax.barh(yp, c["ND-LR higher"], height=0.62, color=FOCUS, edgecolor=DARK, lw=0.5,
                label="ND-LR higher")
        ax.barh(yp, c["comparator higher"], height=0.62, left=c["ND-LR higher"], color=REFERENCE,
                hatch="///", edgecolor=DARK, lw=0.5, label="comparator higher")
        ax.barh(yp, c["tie"], height=0.62, left=c["ND-LR higher"] + c["comparator higher"],
                color=PALE, edgecolor=DARK, lw=0.5, label="exact tie")
        ax.axvline(50, color=DARK, lw=0.7, ls="--", zorder=3)
        ax.set(yticks=yp, yticklabels=args.comparators, xlim=(0, 100),
               xticks=[0, 25, 50, 75, 100], title=METRIC_NAMES[m])
        gridlines(ax)
    axes[0, 0].invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("paired splits (%)")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3)
    save(args, fig, "exp_X4_paired_win_loss")

    code = {c: i for i, c in enumerate(args.classes)}
    gap = (preds[args.tree].map(code) - preds["grade"].map(code)).abs()
    print(f"\nND-LR predictions within one grade, pooled over {len(preds)} held-out predictions: "
          f"{(gap <= 1).mean():.4%}")

    trees = oil.ordered_trees(args.classes)
    sweep = with_nd(args, pd.DataFrame({"model": trees})).assign(tree=trees).set_index("model")
    sweep = sweep.join(mean[["balAcc", "acc", "macroF1"]])
    sweep["sum of ranks"] = sweep[["balAcc", "acc", "macroF1"]].rank(ascending=False).sum(axis=1)
    print("\nordered ND-LR structures")
    print(sweep.set_index("tree").round(3).to_string())

    decomposition = [m for m in ["ND-LR", "LR-OVR", "LR-OVO", "SVM-OVR", "SVM-OVO"] if m in mean.index]
    print("\ndecompositions")
    print(mean.loc[decomposition].round(3).to_string())
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.2), sharey=True)
    for ax, m in zip(axes, oil.METRICS):
        boxes(ax, [per_split.loc[per_split["model"] == k, m] for k in decomposition], decomposition)
        ax.set_title(SHORT_NAMES[m])
    fig.tight_layout()
    save(args, fig, "modelLR_ovo_ovr")


def leakage(args):
    runs = with_nd(args, pd.read_csv(args.out / "leakage_scores.csv"))
    wide = runs.set_index(["arm", "model", "seed"]).sort_index()
    models = list(dict.fromkeys(runs["model"]))
    gaps = []
    for arm in runs["arm"].unique():
        if arm == "honest":
            continue
        for m in models:
            for metric in oil.METRICS:
                gap, lo, hi = mean_ci(wide.loc[(arm, m), metric] - wide.loc[("honest", m), metric])
                gaps.append({"arm": arm, "model": m, "metric": metric, "gap": gap, "lo": lo, "hi": hi})
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(args.out / "leakage_gaps.csv", index=False)
    table = gaps.pivot_table(index=["arm", "model"], columns="metric", values="gap")
    print("leaked minus honest, mean over paired splits")
    print(table[list(oil.METRICS)].round(3).to_string())
    prep = gaps[gaps["arm"] == "prep"]
    print(f"\nlargest preprocessing gap {prep['gap'].abs().max():.3f}, intervals clear of zero:")
    print(prep[(prep["lo"] > 0) | (prep["hi"] < 0)][["model", "metric", "gap"]].round(4).to_string(index=False))
    tuned = gaps[gaps["arm"] == "tuned"].groupby("model")["gap"].max().sort_values(ascending=False)
    print("\nlargest test-tuning gap per model")
    print(tuned.round(4).to_string())
    level = runs.groupby(["arm", "model"])["macroF1"].mean()
    print(f"\nLR macro-F1: engine-grouped {level[('honest', 'LR')]:.3f}, row-level {level[('split', 'LR')]:.3f}")
    print("\nmean macro-F1, honest and with settings picked on the test fold")
    print(level.unstack(0)[["honest", "tuned"]].round(3).to_string())

    arms = [("split", "row-level splits, not engine-grouped", FOURTH, DARK, "D"),
            ("prep", "preprocessing on all rows", REFERENCE, REFERENCE_LINE, "s"),
            ("tuned", "settings picked on the test fold", FOCUS, FOCUS_LINE, "o"),
            ("all", "all three", DARK, DARK, "*")]
    pitch = 1.35
    yp = np.arange(len(models)) * pitch
    offsets = np.linspace(-0.38, 0.38, len(arms))
    drawn = gaps[gaps["arm"].isin([a[0] for a in arms])]
    pad = 0.04 * (drawn["hi"].max() - drawn["lo"].min())
    xlim = (drawn["lo"].min() - pad, drawn["hi"].max() + pad)
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 6.4), sharey=True)
    for ax, metric in zip(axes.ravel(), oil.METRICS):
        for i in range(0, len(models), 2):
            ax.axhspan(yp[i] - pitch / 2, yp[i] + pitch / 2, color=PALE, alpha=0.45, lw=0, zorder=0)
        for (arm, label, fill, line, marker), off in zip(arms, offsets):
            s = gaps[(gaps["metric"] == metric) & (gaps["arm"] == arm)].set_index("model").loc[models]
            ax.errorbar(s["gap"], yp + off, xerr=[s["gap"] - s["lo"], s["hi"] - s["gap"]],
                        fmt=marker, mfc=fill, mec=line, color=line, ecolor=line, capsize=2,
                        ms=7 if marker == "*" else 4.5, lw=0.9, zorder=3,
                        label=label if metric == "acc" else None)
        ax.axvline(0, color=DARK, lw=0.7, ls="--", zorder=1)
        ax.set(yticks=yp, yticklabels=models, ylim=(yp[-1] + pitch / 2, yp[0] - pitch / 2),
               xlim=xlim, title=METRIC_NAMES[metric])
        gridlines(ax)
    for ax in axes[1]:
        ax.set_xlabel("leaked minus honest, paired splits")
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.025))
    save(args, fig, "exp_X5_leakage")


def variability(args):
    runs = with_nd(args, pd.read_csv(args.out / "split_variability.csv"))
    for m, part in runs.groupby("model"):
        rows = []
        for metric in oil.METRICS:
            v = part[metric]
            p5, p95 = np.percentile(v, [5, 95])
            rows.append({"metric": metric, "mean": v.mean(), "min": v.min(), "max": v.max(),
                         "p5": p5, "p95": p95, "p95 - p5": p95 - p5, "max - mean": v.max() - v.mean()})
        print(f"{m}, {part['seed'].nunique()} engine-grouped splits")
        print(pd.DataFrame(rows).set_index("metric").round(4).to_string(), "\n")

        fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2))
        for ax, metric in zip(axes, ["balAcc", "acc", "macroF1"]):
            v = part[metric].to_numpy()
            lo, hi = np.percentile(v, [5, 95])
            ax.hist(v, bins=45, color=FOCUS, edgecolor=FOCUS_LINE, lw=0.2)
            ax.axvspan(lo, hi, color=FOCUS, alpha=.15, lw=0)
            ax.axvline(v.mean(), color=DARK, lw=1.2, ls="--")
            ax.set_title(METRIC_NAMES[metric])
            ax.set_yticks([])
        axes[0].set_ylabel("splits")
        fig.tight_layout()
        save(args, fig, f"exp_X2_split_variability_{m}")


def tex_int(n):
    return f"{int(n):,}".replace(",", "{,}")


def escape(text):
    return "".join(TEX.get(c, c) for c in text)


def quote(note, limit=260):
    """The note as the laboratory wrote it, cut at a word and marked when longer than limit."""
    note = str(note)
    if len(note) <= limit:
        return escape(note)
    return escape(note[:limit].rsplit(" ", 1)[0]) + r" \ldots"


def flags(args):
    res = pd.read_csv(args.data.parent / "grade_review.csv", keep_default_na=False, na_values=[""])
    hit = res[res["flagged"]].sort_values("probability", ascending=False)
    print(f"flagged {len(hit)} of {len(res)} ({len(hit) / len(res):.1%})")
    print("direction:", hit["direction"].value_counts().to_dict())
    print("steps:", hit["steps"].value_counts().sort_index().to_dict())

    lean = (hit.groupby(["analyst", "direction"]).size().unstack(fill_value=0)
            .reindex(columns=["optimistic", "cautious"], fill_value=0))
    lean["flagged"] = lean.sum(axis=1)
    lean["graded"] = res.groupby("analyst").size()
    lean = lean.sort_values("flagged", ascending=False)
    print(lean.to_string())
    fig, ax = plt.subplots(figsize=(WIDE, 3.0))
    yp = np.arange(len(lean))
    ax.barh(yp, lean["cautious"], color=FOCUS, edgecolor=DARK, lw=0.5, label="cautious")
    ax.barh(yp, lean["optimistic"], left=lean["cautious"], color=REFERENCE, hatch="///",
            edgecolor=DARK, lw=0.5, label="optimistic")
    for i, r in enumerate(lean.itertuples()):
        ax.text(r.flagged + 1, i, f"{r.flagged} of {r.graded}", va="center", fontsize=7, color=DARK)
    ax.set_yticks(yp, lean.index)
    ax.invert_yaxis()
    ax.set_xlabel("unanimous disagreements with the model consensus")
    ax.legend(loc="lower right")
    gridlines(ax)
    fig.tight_layout()
    save(args, fig, "modelLR_analyst_lean")

    voters = [c for c in res if c.endswith("_grade")]
    n = NUMBER_WORDS.get(len(voters), str(len(voters)))
    captions = {"optimistic": "Highest-ranked optimistic disagreements between the expert-assigned grade "
                              f"and the {n} models. Optimistic means the expert graded the sample "
                              "healthier than the models.",
                "cautious": "Highest-ranked cautious disagreements between the expert-assigned grade and "
                            "model consensus. Cautious means the expert graded the sample as more "
                            f"severe than the {n} models agreed on."}
    blocks = []
    for direction, caption in captions.items():
        top = hit[hit["direction"] == direction].head(args.rows)
        body = [f"${r.grade} \\rightarrow {r.consensus}$ & {r.steps} & {r.probability:.2f} &\n"
                f"{quote(r.note)} \\\\" for r in top.itertuples()]
        blocks.append("\n".join([
            r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}",
            rf"\label{{tab:{direction}_disagreements}}", r"\begin{tabularx}{\linewidth}{c c c X}",
            r"\hline", r"Expert $\rightarrow$ model & Steps & Mean model probability & Expert note \\",
            r"\hline", *body, r"\hline", r"\end{tabularx}", r"\end{table}"]))
    (args.out / "table_disagreements.tex").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def grading_periods(args, res):
    """Whether the experts graded the same engines over the same period: the quarters in which
    each graded, and how often the grading expert changes along an engine's samples."""
    d, _, g = oil.load(args)
    names = [res["reference"].iloc[0], *[t.removeprefix(args.by + "_")
                                         for t in dict.fromkeys(res["term"])
                                         if t.startswith(args.by + "_")]]
    who = d[args.by]
    shared = set.intersection(*(set(g[who == a]) for a in names))
    part = d[who.isin(names) & d[args.group].isin(shared)]
    quarter = pd.to_datetime(part["date"]).dt.to_period("Q")
    print(pd.crosstab(quarter, part[args.by]).to_string())
    switches = part.groupby(args.group)[args.by].agg(lambda s: int((s != s.shift()).sum()) - 1)
    print(f"grading expert changes along each engine's samples: median {switches.median():.0f}, "
          f"minimum {switches.min()}, over {len(shared)} engines")
    return names, len(part), len(shared)


def analysts(args):
    res = pd.read_csv(args.out / "analyst_effects.csv")
    names, samples, engines = grading_periods(args, res)
    terms = res[res["term"].str.startswith(f"{args.by}_")]
    print(terms.round(3).to_string(index=False))
    columns = list(dict.fromkeys(terms["term"]))
    cell = {(r.node, r.term): (f"{r.coef:.3f}\\ [{r.lo:.3f},\\,{r.hi:.3f}]", r.lo > 0 or r.hi < 0)
            for r in terms.itertuples()}
    body = []
    for node in dict.fromkeys(terms["node"]):
        left, right = node.split("|")
        cells = [f"$\\mathbf{{{text}}}$" if clear else f"${text}$"
                 for text, clear in (cell[(node, t)] for t in columns)]
        grades = [args.classes[i] for i in oil.grade_ids(right, args.classes)]
        right = right if len(grades) == 1 else f"\\{{{','.join(grades)}\\}}"
        body.append(f"${left}$ vs ${right}$ &\n" + " &\n".join(cells) + r" \\")
    header = " & ".join(f"{t.removeprefix(args.by + '_')} vs {names[0]}" for t in columns)
    graded = ", ".join(names[:-1]) + f", or {names[-1]}"
    count = {2: "both", 3: "all three", 4: "all four", 5: "all five", 6: "all six"}
    (args.out / "table_analysts.tex").write_text("\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{Expert coefficients from ND-LR node models fitted to the {tex_int(samples)} "
        rf"samples graded by {graded} on the {engines} engines that "
        rf"{count.get(len(names), f'all {len(names)}')} experts graded, with {names[0]} as the "
        rf"reference. Values are coefficients with 95\% engine-bootstrap intervals "
        rf"({tex_int(res['draws'].iloc[0])} bootstrap resamples). Negative values mean the "
        rf"expert was less likely than {names[0]} to place a sample on the more severe side of "
        r"the node. Bold indicates an interval that does not include zero.}",
        r"\label{tab:analyst_effects}", r"\begin{tabular}{l" + "c" * len(columns) + "}", r"\hline",
        f"Node & {header} \\\\", r"\hline", *body, r"\hline", r"\end{tabular}", r"\end{table}",
    ]) + "\n", encoding="utf-8")


def main():
    ap = oil.parser(__doc__)
    ap.add_argument("command", choices=["data", "model", "compare", "leakage", "variability",
                                        "flags", "analysts", "all"])
    ap.add_argument("--table-models", nargs="+",
                    help="rows after ND-LR in the results table and MCC figure, "
                         "by default every one of LR LDA RF SVM DT XGB MLP in model_comparison.csv")
    ap.add_argument("--comparators", nargs="+",
                    help="models ND-LR is paired against, by default as --table-models, "
                         "the first is also compared per grade")
    ap.add_argument("--rows", type=int, default=4, help="rows in each disagreement table")
    ap.add_argument("--by", default="analyst", help="column naming the expert, used by evaluate.py analysts")
    args = oil.parse_args(ap)
    style()
    steps = {"data": data, "model": model, "compare": compare, "leakage": leakage,
             "variability": variability, "flags": flags, "analysts": analysts}
    for name in steps if args.command == "all" else [args.command]:
        print(f"\n== {name} ==")
        steps[name](args)


if __name__ == "__main__":
    main()
