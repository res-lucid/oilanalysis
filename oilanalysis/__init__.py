"""oilanalysis: grade oil samples with nested dichotomies of logistic regressions (ND-LR) and
other classifiers, evaluated with each engine kept wholly in the training or the test set."""

from .oil import (
    ESTIMATORS,
    METRICS,
    NODE,
    fit_nd,
    fit_tuned,
    inner_folds,
    ordered_trees,
    outer_split,
    parse_tree,
    predict_tuned,
    preprocessor,
    scores,
)

__all__ = [
    "ESTIMATORS",
    "METRICS",
    "NODE",
    "fit_nd",
    "fit_tuned",
    "inner_folds",
    "ordered_trees",
    "outer_split",
    "parse_tree",
    "predict_tuned",
    "preprocessor",
    "scores",
]
