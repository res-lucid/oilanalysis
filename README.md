# oilanalysis

oilanalysis trains and tests classifiers on oil-analysis data with each engine kept wholly in the
training or the test set. It measures how much row-level splitting, preprocessing fitted on all
rows, and settings picked on the test fold raise the scores, and it lists the expert-assigned grades
that a set of models agree should be different, so an expert can review them. Its main model is a
nested dichotomy of logistic regressions (ND-LR), fitted with
[ndscape](https://github.com/res-lucid/ndscape), a Python package for nested dichotomies, which grades a sample through one binary decision
per grade boundary, each with coefficients that can be read directly. It also includes the 2,110
engine-oil samples used in the paper.

## Install

Python 3.11 or later.

```bash
pip install git+https://github.com/res-lucid/oilanalysis
```

This installs ndscape and the other dependencies, and the commands `oilanalysis-evaluate` and
`oilanalysis-report`. To reproduce the paper with the versions it used, clone the repository and
install the pinned versions first:

```bash
git clone https://github.com/res-lucid/oilanalysis
cd oilanalysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Use it on your data

Give a CSV with one row per sample, then name the grade column, the engine column, the grades
healthiest first, the feature columns and the nested dichotomy:

```bash
oilanalysis-evaluate --data my.csv --target condition --group unit --classes N M S \
    --features fe cu si visc --tree "(N(MS))" compare --models LR RF
oilanalysis-report --data my.csv --target condition --group unit --classes N M S \
    --features fe cu si visc --tree "(N(MS))" compare
```

The first command writes `results/model_comparison.csv`. The second prints the mean scores,
paired comparisons, per-grade recall and precision and the share of ND-LR predictions within one
grade, and writes the figures and LaTeX tables.

Feature columns are used as they are: `<1` is read as 0 and any other text as missing, missing
values are median-imputed and the columns standardised inside each training fold, and 0/1
columns are mode-imputed. Rows whose grade is not in `--classes` are left out.

## Use it from Python

```python
import oilanalysis as oa

classes = ["N", "M", "S"]                    # healthiest first, y holds positions 0, 1, 2
tree = oa.parse_tree("(N(MS))", classes)
nd, cutoffs = oa.fit_tuned(X_train, y_train, engines_train, tree)
oa.predict_tuned(nd, cutoffs, X_test)        # ND-LR with tuned cutoffs
nd.predict(X_test)                           # routed by the product of node probabilities
nd.models[tree[0]][-1].coef_                 # coefficients at the N vs MS node
```

`X_train` and `X_test` are numeric arrays without missing values. `oa.preprocessor(frame)` gives
the imputation and scaling described above. `fit_tuned` tunes the cutoffs on engine-grouped inner
folds of the training rows, and `columns=` restricts each node to a list of column positions.
`nd` is an ndscape `ND`, so the rest of ndscape works on it. `oa.outer_split`, `oa.scores` and
`oa.ESTIMATORS` give the paper's splits, metrics and comparison classifiers.

## Commands

`oilanalysis-evaluate` runs one experiment and writes one CSV to `--out` (default `results/`).
`oilanalysis-report` reads those CSVs and prints the numbers, and writes figures to
`results/figures/` and LaTeX tables to `results/`. `python -m oilanalysis.evaluate` and
`python -m oilanalysis.report` do the same.

- `compare`: every nested dichotomy whose splits keep the grades in order, the `--tree` ND-LR
  without tuned cutoffs, and the `--models` classifiers, on 50 engine-grouped splits
  (`model_comparison.csv`)
- `leakage`: the `--models` classifiers and ND-LR on the same splits, honestly and with
  preprocessing fitted on all rows, settings picked on the test fold, row-level splits, and all
  three (`leakage_scores.csv`)
- `variability`: the `--models` classifiers over 1,000 engine-grouped splits
  (`split_variability.csv`)
- `flags`: each engine held out in turn. A sample is flagged when every `--models` voter
  predicts the same grade and it differs from the expert-assigned one (`grade_review.csv`, written
  next to the `--data` file)
- `analysts`: the ND-LR nodes refitted with an indicator for each expert, with intervals from an
  engine bootstrap (`analyst_effects.csv`). `--C inf` refits them without a penalty

`--jobs N` sets the number of worker processes (default 4) and goes before the command, as in
`oilanalysis-evaluate --jobs 8 compare`.

## Options

Grades can be letters, digits or names. Single-character grades are written run together, as
above. Longer grade names are separated by spaces in a tree and by `+` in a node:

```bash
oilanalysis-evaluate --data my.csv --target condition --group unit \
    --classes Normal Caution Critical --features fe cu si visc \
    --drop visc:Normal+Caution+Critical --tree "(Normal (Caution Critical))" compare
```

`--only FEATURE:NODE` offers a feature only at the node over those grades and
`--drop FEATURE:NODE` withholds it there. Without `--features` these default to the paper's rules.
With it they default to none.

`compare`, `variability`, `leakage` and `flags` take any classifier in `ESTIMATORS` in `oil.py`,
or `MLP`, through `--models`, so one line added there adds a classifier. `compare`,
`variability` and `flags` also take trees. In `leakage`, a classifier with no entry in `GRIDS` in
`evaluate.py` is scored at its own setting only. In `flags`, `ND-LR` is the `--tree` ND-LR, and a
classifier without `predict_proba`, such as `SVM`, gives probability 1 to the grade it predicts.

Options that take several values swallow a command written straight after them, so end with a
single-valued option such as `--tree`.

`oilanalysis-report data` needs the oil columns. `oilanalysis-report flags` needs `analyst` and
`note` in `grade_review.csv`, which `oilanalysis-evaluate flags --keep analyst note` copies there.
`analysts` needs a column naming the expert, set with `--by`, and `oilanalysis-report analysts` also
needs `date`.

## Model names

In the CSVs an ND-LR is named by its tree, for example `(A(B(CX)))`. The reports call the
`--tree` one ND-LR. The `--tree` ND-LR followed by ` soft` (`ND-LR soft` in the reports) assigns
each sample the grade with the highest product of node probabilities instead of using tuned
cutoffs. ndscape fits every ND-LR node, each on that node's feature columns, and computes the
soft routing; the tuned cutoffs are applied in `oil.py`. `MLP` chooses its architecture and
penalty on the inner folds and its epoch count on held-out training engines, trains on a
class-balanced oversample, and averages five initialisations. It is the MLP in every experiment.

## The included dataset

`oilanalysis/data/Multiclass_oil_analysis_on_diesel_trucks.csv` has one row per oil sample, 2,110
samples from 63 engines, in engine and date order. `sample` numbers the rows (S0001 to S2110),
`engine` is the engine (E01 to E63), `model` the engine model (MD1 to MD8), `truck` 1 for a haul
truck, `service_interval` the oil change interval in hours (250 or 500), `grade` the
expert-assigned grade (A, B, C, X), `analyst` the anonymised expert (R1 to R6, or Unknown), `note`
the expert's free text, `oilhours`, `meterread` and `oilchanged` the service record, and the remaining columns the laboratory
measurements. Measurements are as the laboratory recorded them: values below detection are `<1`,
which the paper's features and the correlation figure read as 0. Empty cells and the `-`
recorded for some `TBN` values are missing, and are median-imputed inside each training fold.

Customer and site fields are removed, and laboratory sample numbers, engine serial numbers and
model numbers are replaced. In the notes, names and other personal information are removed and
the equipment maker's name is replaced by "the manufacturer". The notes are otherwise as the
laboratory wrote them, spelling and punctuation included, and the disagreement tables print them
unchanged.

## Reproduce the paper

This repository holds the code and data for the paper *Interpretable multiclass classification of
diesel engine oil condition*. Run from the repository
folder with the default options. The results used in the paper are already in `results/`, so
`oilanalysis-report all` works straight away. To recompute them:

```bash
oilanalysis-evaluate compare                # model_comparison.csv, about 10 min
oilanalysis-evaluate --jobs 10 leakage      # leakage_scores.csv, about 3 h
oilanalysis-evaluate variability            # split_variability.csv, under a minute
oilanalysis-evaluate flags                  # grade_review.csv in oilanalysis/data, about 15 min
oilanalysis-evaluate analysts               # analyst_effects.csv, under a minute
oilanalysis-report all
```

Times are for a 16-core machine, with the default 4 workers unless `--jobs` is given. `leakage` is
slow because every MLP setting in its test-fold search is fitted five times, as the reported MLP
is. It prints libsvm "Solver terminated early" warnings: its SVM grid caps iterations because
some polynomial-kernel settings never converge.

The paper's defaults: `compare` runs LR, LDA, RF, SVM, DT, XGB, MLP and the one-vs-rest and
one-vs-one decompositions of LR and SVM. `leakage` runs LR, LDA, RF, SVM, DT, XGB and MLP with
ND-LR. `flags` uses ND-LR routed softly, LR, RF, XGB and MLP, each fitted without the held-out
engine and without the previous-grade features. `analysts` refits the ND-LR nodes on the
samples graded by R1, R2 or R3 on the engines all three graded, with an indicator for each
expert after the first.

`oilanalysis-report all` prints the numbers the paper quotes and writes:

| Output | Paper |
| --- | --- |
| `figures/modelLR_feature_correlation`, `figures/modelLR_feature_dists` | Section 3 and 4.1 figures |
| `table_results.tex`, `figures/modelLR_nd_vs_flat_mcc` | results table and MCC figure |
| `figures/exp_X4_paired_win_loss`, `table_grades.tex` | paired comparison figure and per-grade table |
| `figures/exp_X2_split_variability_LR` | split variability figure |
| `figures/exp_X5_leakage` | leakage figure |
| `figures/modelLR_nd_roc`, `figures/modelLR_coefs_A-B-C` | ND-LR ROC and coefficient figures |
| `figures/modelLR_analyst_lean`, `table_disagreements.tex`, `table_analysts.tex` | grade review figure and tables |
| `figures/modelLR_ovo_ovr` | appendix decomposition figure |

Each part can be run alone: `oilanalysis-report data`, `model`, `compare`, `leakage`,
`variability`, `flags` or `analysts`.

## Files

- `oilanalysis/oil.py`: data loading, the paper's features, ND-LR fitting with ndscape, cutoff
  routing, classifiers, splits
- `oilanalysis/evaluate.py`: the experiments
- `oilanalysis/report.py`: figures, LaTeX tables and printed numbers
- `oilanalysis/data/Multiclass_oil_analysis_on_diesel_trucks.csv`: the included dataset
- `oilanalysis/data/grade_review.csv`: the grades the `flags` command flagged, with the experts' notes
- `oilanalysis/fonts/`: Latin Modern, so figure text matches the paper
- `results/`: the paper's results, figures and tables

## Licence

The code is under the MIT licence (`LICENSE`). The data in `oilanalysis/data/` and the results in
`results/` are under the Creative Commons Attribution 4.0 International licence
(`LICENSE-DATA`). The fonts in `oilanalysis/fonts/` are the Latin Modern fonts, distributed under
the GUST Font License.
