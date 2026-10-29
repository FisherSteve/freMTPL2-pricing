"""Identifiable exposure-weighted frequency candidates for the bounded comparison.

The GAM is an explicit P-spline Poisson model, not a wrapper around pyGAM.
Its second-difference penalty is on the *sum* Poisson objective; sklearn's GLM
ridge penalty is on a weight-normalized objective. Their tuning values do not
have the same numerical interpretation. All learned quantities use fit rows.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

from src.data import DataContractError


FEATURES = ("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region")
FACTORS = ("VehPower", "VehBrand", "VehGas", "Area", "Region")
SMOOTHS = ("DrivAge", "VehAge", "BonusMalus", "logDensity")
AGE_BREAKS = (25, 35, 45, 55, 65, 75)
VEHICLE_BREAKS = (1, 6, 11, 21)
BM_BREAKS = (51, 76, 101, 151)


def prospective_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Reject accidental outcomes/IDs instead of silently dropping them."""
    if not isinstance(X, pd.DataFrame) or set(X.columns) != set(FEATURES):
        raise DataContractError("Frequency estimator needs exactly the prospective features")
    if X.empty or X.isna().any().any():
        raise DataContractError("Frequency features must be nonempty and nonmissing")
    out = X.loc[:, FEATURES].copy()
    numeric = out[["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, -1] <= 0).any():
        raise DataContractError("Frequency features require finite values and positive density")
    out["logDensity"] = np.log(out["Density"].to_numpy(dtype=float))
    return out


def response(y, sample_weight, size):
    y = np.asarray(y, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if (y.shape != (size,) or weights.shape != (size,) or not np.isfinite(y).all()
            or not np.isfinite(weights).all() or (y < 0).any() or (weights <= 0).any()
            or np.dot(y, weights) <= 0):
        raise DataContractError("Positive exposures and a nonzero finite claim-rate response required")
    return y, weights


def category_levels(frame, columns):
    return {c: sorted(frame[c].astype(str).unique().tolist()) for c in columns}


def factor_matrix(frame, levels):
    """Reference coding; unseen levels deliberately use the training reference."""
    columns, names = [], []
    for name, values in levels.items():
        observed = frame[name].astype(str).to_numpy()
        for value in values[1:]:
            columns.append((observed == value).astype(float))
            names.append(f"{name}={value}")
    return (np.column_stack(columns) if columns else np.empty((len(frame), 0))), names


def unknown_counts(frame, levels):
    return {c: int((~frame[c].astype(str).isin(values)).sum()) for c, values in levels.items()}


def gram_diagnostics(matrix, weights):
    # A p x p crossproduct avoids a second n x p matrix/SVD on portfolio rows.
    gram = matrix.T @ (weights[:, None] * matrix) / weights.sum()
    eigenvalues = linalg.eigvalsh(gram)
    tolerance = max(float(eigenvalues[-1]), 1.0) * 1e-11
    rank = int((eigenvalues > tolerance).sum())
    condition = float(np.sqrt(eigenvalues[-1] / eigenvalues[0])) if eigenvalues[0] > tolerance else None
    return gram, {"design_columns": matrix.shape[1], "weighted_design_rank": rank,
                  "weighted_design_condition": condition, "rank_tolerance": tolerance}


class FrequencyGLM(RegressorMixin, BaseEstimator):
    """Actuarial binned Poisson GLM with optional coarse age x BM interaction."""

    def __init__(self, alpha=0.0, encoding="additive", max_iter=100, tol=1e-8):
        self.alpha, self.encoding, self.max_iter, self.tol = alpha, encoding, max_iter, tol

    @staticmethod
    def _groups(frame):
        frame = frame.copy()
        for name, original, breaks in (("age_band", "DrivAge", AGE_BREAKS),
                                      ("vehicle_band", "VehAge", VEHICLE_BREAKS),
                                      ("bm_band", "BonusMalus", BM_BREAKS)):
            frame[name] = np.searchsorted(breaks, frame[original].to_numpy(), side="right").astype(str)
        return frame

    def _design(self, frame):
        factors, names = factor_matrix(frame, self.levels_)
        density = (frame["logDensity"].to_numpy() - self.density_mean_) / self.density_scale_
        parts = [np.ones((len(frame), 1)), factors, density[:, None]]
        names = ["intercept", *names, "logDensity_standardized"]
        if self.encoding == "age_bm_interaction":
            age, age_names = factor_matrix(frame, {"age_band": self.levels_["age_band"]})
            bm, bm_names = factor_matrix(frame, {"bm_band": self.levels_["bm_band"]})
            for j, age_name in enumerate(age_names):
                for k, bm_name in enumerate(bm_names):
                    parts.append((age[:, j] * bm[:, k])[:, None])
                    names.append(f"{age_name}:{bm_name}")
        return np.column_stack(parts), names

    def fit(self, X, y, sample_weight):
        if self.encoding not in ("additive", "age_bm_interaction") or self.alpha < 0:
            raise DataContractError("Unregistered GLM encoding/negative penalty")
        frame = self._groups(prospective_frame(X))
        y, weights = response(y, sample_weight, len(frame))
        self.levels_ = category_levels(frame, (*FACTORS, "age_band", "vehicle_band", "bm_band"))
        self.density_mean_ = float(np.average(frame.logDensity, weights=weights))
        self.density_scale_ = float(np.sqrt(np.average((frame.logDensity - self.density_mean_) ** 2, weights=weights)))
        self.density_scale_ = self.density_scale_ or 1.0
        design, names = self._design(frame)
        with threadpool_limits(limits=2):
            gram, diagnostics = gram_diagnostics(design, weights)
            # Empty or unsupported interaction cells are determined by covariates,
            # never outcomes. Retain a deterministic independent column subset.
            _, triangular, pivots = linalg.qr(gram, mode="economic", pivoting=True)
            rank = int((np.abs(np.diag(triangular)) > diagnostics["rank_tolerance"]).sum())
            retained = sorted(int(p) for p in pivots[:rank])
            if 0 not in retained:
                raise DataContractError("Outcome-free GLM rank reduction lost the intercept")
            self.keep_ = np.asarray(retained)
            self.feature_names_ = [names[i] for i in retained]
            reduced = design[:, self.keep_]
            _, reduced_diagnostics = gram_diagnostics(reduced, weights)
            if reduced_diagnostics["weighted_design_rank"] != reduced.shape[1]:
                raise DataContractError("GLM reduced design is still rank deficient")
            # fit_intercept keeps the intercept unpenalized, as required for
            # portfolio total calibration. Column 0 is the explicit constant.
            self.model_ = PoissonRegressor(alpha=self.alpha, fit_intercept=True,
                                          solver="newton-cholesky", max_iter=self.max_iter, tol=self.tol)
            self.model_.fit(reduced[:, 1:], y, sample_weight=weights)
        self.diagnostics_ = {**diagnostics, "retained_design": reduced_diagnostics,
            "dropped_covariate_columns": [name for i, name in enumerate(names) if i not in retained],
            "feature_names": self.feature_names_, "category_levels": self.levels_,
            "category_reference": {c: levels[0] for c, levels in self.levels_.items()},
            "density_training_weighted_mean": self.density_mean_, "density_training_weighted_scale": self.density_scale_,
            "iterations": int(self.model_.n_iter_), "converged": bool(self.model_.n_iter_ < self.max_iter),
            "alpha": self.alpha, "encoding": self.encoding,
            "objective": "sum(e*(exp(eta)-y*eta))/sum(e) + alpha/2*sum(nonintercept_coef**2)",
            "offset_equivalence": "At alpha=0, rate response N/e weighted by e has the same score equations as counts with log(e) offset."}
        if not self.diagnostics_["converged"]:
            raise DataContractError("GLM reached its iteration limit")
        return self

    def predict(self, X):
        frame = self._groups(prospective_frame(X))
        design, _ = self._design(frame)
        self.last_unknown_categories_ = unknown_counts(frame, self.levels_)
        return self.model_.predict(design[:, self.keep_][:, 1:])


class FrequencyGAM(RegressorMixin, BaseEstimator):
    """Cubic P-spline GAM with centered smooths and unpenalized factor effects.

    For each full basis B (k columns), weighted-center B, set the last full
    coefficient to zero, and retain k-1 columns. Since B sums to one, this removes
    precisely the constant ambiguity. The penalty is D2[:, :-1]'D2[:, :-1].
    Centering shifts only the intercept and leaves that difference penalty
    unchanged. Uniform training-range knots and constant extrapolation are used.
    """

    def __init__(self, n_splines=10, lam=1.0, max_iter=100, tol=1e-7):
        self.n_splines, self.lam, self.max_iter, self.tol = n_splines, lam, max_iter, tol

    def _design(self, frame):
        factors, names = factor_matrix(frame, self.levels_)
        parts = [np.ones((len(frame), 1)), factors]
        names = ["intercept", *names]
        for feature, transformer in self.splines_.items():
            block = transformer.transform(frame[[feature]].to_numpy())[:, :-1] - self.centers_[feature]
            parts.append(block)
            names.extend(f"s({feature})[{i}]" for i in range(block.shape[1]))
        return np.column_stack(parts), names

    def fit(self, X, y, sample_weight):
        if self.n_splines < 5 or self.lam <= 0:
            raise DataContractError("GAM needs at least 5 basis functions and positive smoothing")
        frame = prospective_frame(X)
        y, weights = response(y, sample_weight, len(frame))
        self.levels_ = category_levels(frame, FACTORS)
        self.splines_, self.centers_, self.slices_ = {}, {}, {}
        factors, _ = factor_matrix(frame, self.levels_)
        start = 1 + factors.shape[1]
        for feature in SMOOTHS:
            values = frame[[feature]].to_numpy(dtype=float)
            if float(np.ptp(values)) == 0:
                continue
            transformer = SplineTransformer(n_knots=self.n_splines - 2, degree=3,
                knots="uniform", include_bias=True, extrapolation="constant")
            basis = transformer.fit_transform(values)
            self.splines_[feature] = transformer
            self.centers_[feature] = np.average(basis[:, :-1], weights=weights, axis=0)
            self.slices_[feature] = slice(start, start + self.n_splines - 1)
            start += self.n_splines - 1
        design, self.feature_names_ = self._design(frame)
        penalty = np.zeros((design.shape[1], design.shape[1]))
        difference = np.diff(np.eye(self.n_splines), n=2, axis=0)[:, :-1]
        for index in self.slices_.values():
            penalty[index, index] = self.lam * difference.T @ difference
        self.penalty_ = penalty
        coefficients = np.zeros(design.shape[1])
        coefficients[0] = np.log(np.dot(weights, y) / weights.sum())
        counts = weights * y

        def objective(beta):
            eta = design @ beta
            with np.errstate(over="ignore"):
                value = np.dot(weights, np.exp(eta)) - np.dot(counts, eta) + .5 * beta @ penalty @ beta
            return float(value)

        converged = False
        with threadpool_limits(limits=2):
            _, rank = gram_diagnostics(design, weights)
            history = [objective(coefficients)]
            for iteration in range(1, self.max_iter + 1):
                eta = design @ coefficients
                mu = weights * np.exp(eta)
                gradient = design.T @ (mu - counts) + penalty @ coefficients
                gradient_norm = float(np.max(np.abs(gradient)) / weights.sum())
                if gradient_norm <= self.tol:
                    converged = True
                    break
                hessian = design.T @ (mu[:, None] * design) + penalty
                try:
                    step = linalg.solve(hessian, gradient, assume_a="pos")
                except linalg.LinAlgError as error:
                    raise DataContractError("GAM penalized design is not identifiable") from error
                decrement = float(gradient @ step)
                fraction = 1.0
                while fraction >= 2 ** -25:
                    candidate = coefficients - fraction * step
                    value = objective(candidate)
                    if np.isfinite(value) and value <= history[-1] - 1e-4 * fraction * decrement:
                        coefficients = candidate
                        history.append(value)
                        break
                    fraction /= 2
                else:
                    raise DataContractError("GAM Newton line search failed before convergence")
            if not converged:
                raise DataContractError("GAM did not converge within the registered iteration limit")
            self.coef_ = coefficients
            mu = weights * np.exp(design @ coefficients)
            data_hessian = design.T @ (mu[:, None] * design)
            total_hessian = data_hessian + penalty
            eigenvalues = linalg.eigvalsh(total_hessian)
            if eigenvalues[0] <= 0:
                raise DataContractError("GAM penalized Hessian is not positive definite")
            influence = linalg.solve(total_hessian, data_hessian, assume_a="pos")
        self.diagnostics_ = {**rank, "feature_names": self.feature_names_,
            "category_levels": self.levels_, "category_reference": {c: v[0] for c, v in self.levels_.items()},
            "iterations": iteration - 1, "converged": converged, "normalized_gradient_max": gradient_norm,
            "n_splines": self.n_splines, "lambda": self.lam, "degree": 3,
            "effective_degrees_of_freedom": float(np.trace(influence)),
            "smooth_edf": {c: float(np.diag(influence)[index].sum()) for c, index in self.slices_.items()},
            "penalized_hessian_condition": float(eigenvalues[-1] / eigenvalues[0]),
            "objective": "sum(e*(exp(eta)-y*eta)) + lambda/2*sum_smooths(sum(diff(full_beta,2)**2))",
            "objective_history": history, "identifiability": "exposure-centered full B-spline basis; last full coefficient fixed at zero; factor reference omitted",
            "smooth_ranges": {c: [float(frame[c].min()), float(frame[c].max())] for c in self.splines_},
            "knots": {c: transformer.bsplines_[0].t.tolist() for c, transformer in self.splines_.items()},
            "extrapolation": "constant outside each training range",
            "intervals": "No p-values or confidence bands computed; EDF conditional on fixed lambda."}
        return self

    def predict(self, X):
        frame = prospective_frame(X)
        self.last_unknown_categories_ = unknown_counts(frame, self.levels_)
        design, _ = self._design(frame)
        return np.exp(design @ self.coef_)


class FrequencyHistogram(RegressorMixin, BaseEstimator):
    """One CPU histogram Poisson benchmark with train-only native categories."""

    def __init__(self, max_leaf_nodes=15, l2_regularization=1.0, max_iter=300,
                 learning_rate=.05, random_state=20261029):
        self.max_leaf_nodes, self.l2_regularization, self.max_iter = max_leaf_nodes, l2_regularization, max_iter
        self.learning_rate, self.random_state = learning_rate, random_state

    def _design(self, frame):
        numeric = frame[["VehPower", "VehAge", "DrivAge", "BonusMalus", "logDensity"]].to_numpy(dtype=float)
        codes = []
        for feature, levels in self.levels_.items():
            mapping = {value: i for i, value in enumerate(levels)}
            codes.append(frame[feature].astype(str).map(mapping).to_numpy(dtype=float, na_value=np.nan))
        return np.column_stack([numeric, *codes])

    def fit(self, X, y, sample_weight):
        frame = prospective_frame(X)
        y, weights = response(y, sample_weight, len(frame))
        self.levels_ = category_levels(frame, ("VehBrand", "VehGas", "Area", "Region"))
        self.model_ = HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization, max_iter=self.max_iter,
            learning_rate=self.learning_rate, early_stopping=False, random_state=self.random_state,
            categorical_features=[False] * 5 + [True] * 4)
        with threadpool_limits(limits=2):
            self.model_.fit(self._design(frame), y, sample_weight=weights)
        self.diagnostics_ = {"converged": True, "iterations": int(self.model_.n_iter_),
            "convergence_meaning": "Fixed registered boosting iteration budget; not an optimizer convergence claim",
            "early_stopping": False, "category_levels": self.levels_, "unknown_category_rule": "missing-value branch learned on training data",
            "feature_names": ["VehPower", "VehAge", "DrivAge", "BonusMalus", "logDensity", *self.levels_],
            "parameters": self.model_.get_params(), "threads": 2}
        return self

    def predict(self, X):
        frame = prospective_frame(X)
        self.last_unknown_categories_ = unknown_counts(frame, self.levels_)
        with threadpool_limits(limits=2):
            return self.model_.predict(self._design(frame))


def make_estimator(configuration: dict):
    """Construct one registered candidate; configuration contains family + params."""
    family = configuration["family"]
    parameters = configuration.get("parameters", {})
    if family == "glm":
        return FrequencyGLM(**parameters)
    if family == "gam":
        return FrequencyGAM(**parameters)
    if family in ("histogram_poisson", "histogram"):
        return FrequencyHistogram(**parameters)
    if family == "intercept":
        from src.validation import ExposureIntercept
        return ExposureIntercept()
    raise DataContractError(f"Unknown frequency model family: {family}")
