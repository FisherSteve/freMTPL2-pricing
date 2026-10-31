"""Claim-row severity candidates with prospective policy features only.

Each positive recorded claim receives unit weight. Exposure, realized claim
counts and loss amounts are not feature columns. The same fitted prediction
rule applies to every policy, including policies without a realized claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import GammaRegressor
from threadpoolctl import threadpool_limits

from src.data import DataContractError


FEATURES = ("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")
FACTORS = ("VehPower", "VehBrand", "VehGas", "Area", "Region", "age_band", "vehicle_band", "bm_band")
AGE_BREAKS = (25, 35, 45, 55, 65, 75)
VEHICLE_BREAKS = (1, 6, 11, 21)
BM_BREAKS = (51, 76, 101, 151)


def prospective_frame(X):
    if not isinstance(X, pd.DataFrame) or set(X.columns) != set(FEATURES) or len(X.columns) != len(FEATURES):
        raise DataContractError("Severity requires exactly the unique prospective feature columns")
    if X.empty or X.isna().any().any():
        raise DataContractError("Severity features must be nonempty and nonmissing")
    frame = X.loc[:, list(FEATURES)].copy()
    numeric = frame[["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, -1] <= 0).any():
        raise DataContractError("Severity covariates need finite values and positive Density")
    frame["logDensity"] = np.log(numeric[:, -1])
    for name, original, breaks in (("age_band", "DrivAge", AGE_BREAKS),
                                   ("vehicle_band", "VehAge", VEHICLE_BREAKS),
                                   ("bm_band", "BonusMalus", BM_BREAKS)):
        frame[name] = np.searchsorted(breaks, frame[original].to_numpy(), side="right").astype(str)
    return frame


def positive_claims(y, sample_weight, size):
    if sample_weight is not None:
        raise DataContractError("Severity uses unit claim-row weights; do not pass exposure or policy weights")
    values = np.asarray(y, dtype=float)
    if values.shape != (size,) or not np.isfinite(values).all() or (values <= 0).any():
        raise DataContractError("Severity response must be finite, strictly positive claim amounts")
    return values


def positive_predictions(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataContractError("Severity predictions must be finite positive EUR per claim")
    return values


def training_totals(values, fitted):
    observed = float(values.sum())
    predicted = float(fitted.sum())
    largest = np.sort(values)[::-1]
    return {"training_claim_rows": len(values), "training_loss_eur": observed,
        "training_predicted_claim_eur_sum": predicted, "training_actual_over_predicted": observed / predicted,
        "largest_training_claim_share": float(largest[0] / observed),
        "largest_ceil_one_percent_training_claim_share": float(largest[:max(1, int(np.ceil(.01 * len(values))))].sum() / observed),
        "units": "EUR per positive recorded claim; unit weight per claim row; no exposure offset"}


class SeverityDesign(TransformerMixin, BaseEstimator):
    """Fixed actuarial groups, training-reference factors and full-rank design."""

    def _matrix(self, frame):
        columns, names = [np.ones(len(frame))], ["intercept"]
        for feature, levels in self.levels_.items():
            observed = frame[feature].astype(str).to_numpy()
            for value in levels[1:]:
                columns.append((observed == value).astype(float))
                names.append(f"{feature}={value}")
        columns.append((frame.logDensity.to_numpy() - self.density_mean_) / self.density_scale_)
        names.append("logDensity_standardized")
        return np.column_stack(columns), names

    def fit(self, X, y=None):
        # The transformation never reads y, even when called through a pipeline.
        frame = prospective_frame(X)
        self.levels_ = {c: sorted(frame[c].astype(str).unique().tolist()) for c in FACTORS}
        self.density_mean_ = float(frame.logDensity.mean())
        self.density_scale_ = float(frame.logDensity.std(ddof=0)) or 1.0
        matrix, names = self._matrix(frame)
        with threadpool_limits(limits=2):
            # Preserve the intercept, then identify a covariate-only independent
            # subset from the centered Gram matrix. This also handles constant
            # density and exact categorical aliases without using claim amounts.
            centered = matrix[:, 1:] - matrix[:, 1:].mean(axis=0)
            gram = centered.T @ centered / len(matrix)
            _, triangular, pivots = linalg.qr(gram, pivoting=True, mode="economic")
            tolerance = max(float(np.max(np.abs(np.diag(triangular)), initial=0)), 1.0) * 1e-11
            rank = int((np.abs(np.diag(triangular)) > tolerance).sum())
            self.keep_ = np.asarray([0, *sorted(int(i) + 1 for i in pivots[:rank])])
            reduced = matrix[:, self.keep_]
            q, triangular = linalg.qr(reduced, mode="economic")
            if len(reduced) < reduced.shape[1] or np.linalg.matrix_rank(triangular) != reduced.shape[1]:
                raise DataContractError("Severity design is not identifiable after covariate-only reduction")
            self.training_leverage_ = np.square(q).sum(axis=1)
            singular = linalg.svdvals(triangular)
        self.feature_names_ = [names[i] for i in self.keep_]
        self.diagnostics_ = {"original_design_columns": matrix.shape[1], "retained_design_columns": len(self.keep_),
            "retained_design_rank": len(self.keep_), "design_condition": float(singular[0] / singular[-1]),
            "dropped_covariate_columns": [name for i, name in enumerate(names) if i not in self.keep_],
            "feature_names": self.feature_names_, "category_levels": self.levels_,
            "category_reference": {c: levels[0] for c, levels in self.levels_.items()},
            "density_training_claim_mean": self.density_mean_, "density_training_claim_scale": self.density_scale_,
            "unknown_category_rule": "All-zero factor indicators, equivalent to the corresponding training reference",
            "maximum_training_design_leverage": float(self.training_leverage_.max()),
            "sum_training_design_leverage": float(self.training_leverage_.sum()),
            "leverage_definition": "diag(X @ inv(X.T @ X) @ X.T); OLS design leverage / Gamma log-link expected-Fisher leverage, not observed-Hessian influence"}
        return self

    def transform(self, X):
        frame = prospective_frame(X)
        self.last_unknown_categories_ = {c: int((~frame[c].astype(str).isin(levels)).sum())
                                         for c, levels in self.levels_.items()}
        matrix, _ = self._matrix(frame)
        return matrix[:, self.keep_]


class ClaimMeanSeverity(RegressorMixin, BaseEstimator):
    """Arithmetic mean of training claim amounts, including every claim row."""

    def fit(self, X, y, sample_weight=None):
        frame = prospective_frame(X)
        values = positive_claims(y, sample_weight, len(frame))
        self.mean_ = float(values.mean())
        self.training_leverage_ = np.full(len(values), 1 / len(values))
        self.diagnostics_ = {"family": "mean", "converged": True,
            "arithmetic_claim_mean_eur": self.mean_, "training_design_rank": 1,
            **training_totals(values, np.full(len(values), self.mean_))}
        return self

    def predict(self, X):
        return np.full(len(prospective_frame(X)), self.mean_)


class GammaSeverity(RegressorMixin, BaseEstimator):
    """Unpenalized Gamma log-link claim mean; no exposure offset or weights."""

    def __init__(self, alpha=0.0, max_iter=200, tol=1e-8):
        self.alpha, self.max_iter, self.tol = alpha, max_iter, tol

    def fit(self, X, y, sample_weight=None):
        frame = prospective_frame(X)
        values = positive_claims(y, sample_weight, len(frame))
        if self.alpha != 0:
            raise DataContractError("S5 registers only the unpenalized Gamma specification")
        self.design_ = SeverityDesign().fit(X)
        matrix = self.design_.transform(X)
        self.model_ = None
        if matrix.shape[1] == 1:
            # Exact intercept-only Gamma MLE; sklearn's array validator requires
            # at least one nonintercept column even when none is identifiable.
            self.coefficients_ = np.array([np.log(values.mean())])
            fitted = np.full(len(values), values.mean())
            iterations = 0
        else:
            self.model_ = GammaRegressor(alpha=self.alpha, fit_intercept=True,
                solver="newton-cholesky", max_iter=self.max_iter, tol=self.tol)
            with threadpool_limits(limits=2):
                self.model_.fit(matrix[:, 1:], values)
                fitted = positive_predictions(self.model_.predict(matrix[:, 1:]))
            self.coefficients_ = np.r_[self.model_.intercept_, self.model_.coef_]
            iterations = int(self.model_.n_iter_)
        if iterations >= self.max_iter:
            raise DataContractError("Gamma severity reached the registered iteration limit")
        self.training_leverage_ = self.design_.training_leverage_.copy()
        self.training_relative_residual_ = values / fitted - 1
        rank = matrix.shape[1]
        residual_df = len(values) - rank
        self.diagnostics_ = {"family": "gamma", "converged": True, "iterations": iterations,
            "alpha": self.alpha, "design": self.design_.diagnostics_, "coefficients": self.coefficients_.tolist(),
            "mean_relative_residual": float(self.training_relative_residual_.mean()),
            "normalized_score_max": float(np.max(np.abs(matrix.T @ self.training_relative_residual_)) / len(values)),
            "pearson_dispersion": float(np.square(self.training_relative_residual_).sum() / residual_df) if residual_df > 0 else None,
            "pearson_dispersion_scope": "Descriptive sum((Y/mu-1)^2)/(n-rank), under Gamma variance phi*mu^2; no distributional fit or calibrated interval claim",
            "objective": "mean(Y/exp(eta) + eta), equivalently half mean Gamma deviance up to constants",
            "calibration": "The intercept score enforces mean(Y/mu)=1, not sum(Y)=sum(mu); no post-fit recalibration",
            **training_totals(values, fitted)}
        return self

    def predict(self, X):
        matrix = self.design_.transform(X)
        self.last_unknown_categories_ = self.design_.last_unknown_categories_
        return positive_predictions(np.exp(matrix @ self.coefficients_))


class LogSmearingSeverity(RegressorMixin, BaseEstimator):
    """OLS on log(Y), retransformed using training-only global Duan smearing.

    E[Y|x] = exp(eta(x))*E[exp(epsilon)|x]. Replacing the second factor with a
    global training residual mean is a modeling approximation when it varies
    with x. Neither the Gaussian error distribution nor conditional unbiasedness
    is claimed. There is no log1p transformation or subtraction of one.
    """

    def fit(self, X, y, sample_weight=None):
        frame = prospective_frame(X)
        values = positive_claims(y, sample_weight, len(frame))
        self.design_ = SeverityDesign().fit(X)
        matrix = self.design_.transform(X)
        log_y = np.log(values)
        with threadpool_limits(limits=2):
            self.coefficients_, _, rank, singular = linalg.lstsq(matrix, log_y, lapack_driver="gelsd")
        if rank != matrix.shape[1]:
            raise DataContractError("Log-severity fit has a rank-deficient design")
        fitted_log = matrix @ self.coefficients_
        self.training_log_residual_ = log_y - fitted_log
        exp_residual = np.exp(self.training_log_residual_)
        self.smearing_ = float(exp_residual.mean())
        fitted = positive_predictions(np.exp(fitted_log) * self.smearing_)
        self.training_leverage_ = self.design_.training_leverage_.copy()
        groups = []
        for feature in ("age_band", "bm_band", "VehGas"):
            for label, indices in frame.groupby(feature, sort=True, observed=True).indices.items():
                r = self.training_log_residual_[indices]
                e = exp_residual[indices]
                groups.append({"feature": feature, "group": str(label), "claim_rows": len(indices),
                    "mean_log_residual": float(r.mean()), "log_residual_variance": float(r.var(ddof=1)) if len(r) > 1 else None,
                    "mean_exp_residual": float(e.mean()), "ratio_to_global_smearing": float(e.mean() / self.smearing_),
                    "largest_exp_residual_share_in_group": float(e.max() / e.sum()),
                    "scope": "Training diagnostic only; group smearing is not used for prediction"})
        self.diagnostics_ = {"family": "lognormal", "converged": True, "design": self.design_.diagnostics_,
            "coefficients": self.coefficients_.tolist(), "response_transform": "log(Y)",
            "smearing_factor": self.smearing_, "smearing_definition": "mean(exp(log(Y_train)-eta_train)); unit claim-row weights",
            "smearing_limits": "Global residual mean need not equal E[exp(epsilon)|x]; no conditional unbiasedness guarantee",
            "smearing_groups": groups, "largest_exp_residual_share": float(exp_residual.max() / exp_residual.sum()),
            "log_residual_variance": float(self.training_log_residual_.var(ddof=1)) if len(values) > 1 else None,
            "maximum_absolute_log_normal_equation": float(np.max(np.abs(matrix.T @ self.training_log_residual_))),
            "fitted_log_min": float(fitted_log.min()), "fitted_log_max": float(fitted_log.max()),
            "objective": "sum((log(Y)-eta)^2); final EUR prediction exp(eta)*global_training_smearing",
            **training_totals(values, fitted)}
        return self

    def predict_log(self, X):
        matrix = self.design_.transform(X)
        self.last_unknown_categories_ = self.design_.last_unknown_categories_
        return matrix @ self.coefficients_

    def predict(self, X):
        return positive_predictions(np.exp(self.predict_log(X)) * self.smearing_)


def make_severity_estimator(configuration):
    family = configuration["family"]
    parameters = configuration.get("parameters", {})
    if family == "mean":
        return ClaimMeanSeverity(**parameters)
    if family == "gamma":
        return GammaSeverity(**parameters)
    if family == "lognormal":
        return LogSmearingSeverity(**parameters)
    raise DataContractError(f"Unknown severity family: {family}")
