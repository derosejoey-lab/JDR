"""
model_engine.py
===============
PCA + Gaussian Hidden Markov Model engine for market regime detection.

Architecture
------------
1. StandardScaler-normalized features (from data_engine) are projected
   through PCA to remove multicollinearity before HMM fitting.
2. A GaussianHMM with a diagonally dominant transmat_prior enforces
   regime persistence, preventing spurious one-day state flips.
3. Hidden states are decoded via the Viterbi algorithm and then
   semantically labelled BULL / BEAR / NEUTRAL by their mean log-return.

Public API
----------
RegimeDetectionModel
    .fit_predict(feature_df, scaled_matrix) -> pd.Series[str]
    .get_transition_matrix()                -> pd.DataFrame
    .get_state_statistics(feature_df, labeled_series) -> pd.DataFrame
    .summary()                              -> dict
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Regime semantic labels in display order
_ORDERED_LABELS = ["BULL", "NEUTRAL", "BEAR"]

# Annualisation factors
_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# RegimeDetectionModel
# ---------------------------------------------------------------------------

class RegimeDetectionModel:
    """
    PCA-compressed Gaussian HMM for market regime detection.

    Parameters
    ----------
    n_states : int
        Number of hidden regimes.  Default 3 (Bull / Neutral / Bear).
    n_pca_components : int
        Number of principal components retained before HMM fitting.
        Default 3; reduces multicollinearity in the 6-feature input.
    n_iter : int
        Maximum Baum-Welch (EM) iterations.  Default 200.
    tol : float
        Convergence tolerance for EM.  Stops when log-likelihood
        improvement falls below this value.  Default 1e-4.
    random_state : int
        Seed for reproducibility.  Default 42.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_pca_components: int = 3,
        n_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_pca_components = n_pca_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Components populated during fit_predict
        self.pca: PCA = PCA(
            n_components=n_pca_components,
            random_state=random_state,
        )
        self.hmm: Optional[GaussianHMM] = None
        self.state_map: dict[int, str] = {}    # raw int  -> semantic label
        self.reverse_map: dict[str, int] = {}  # semantic -> raw int
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_transmat_prior(self) -> np.ndarray:
        """
        Construct a diagonally dominant Dirichlet prior for the transition
        matrix.

        Diagonal elements = 10.0 (strong self-transition prior)
        Off-diagonal elements = 1.0 (weak cross-transition prior)

        In the Baum-Welch M-step, hmmlearn computes:
            transmat_ = max(transmat_prior - 1 + expected_counts, 0)
        so the effective pseudo-count advantage of staying in the same
        state is 10 - 1 = 9.  For a typical financial time-series with
        ~252 observations per year this is sufficient to enforce regimes
        lasting ~3 months.
        """
        prior = np.ones((self.n_states, self.n_states), dtype=float)
        np.fill_diagonal(prior, 10.0)
        return prior

    def _build_init_transmat(self) -> np.ndarray:
        """
        Row-normalised version of the prior used as the starting point
        for the transition matrix before EM begins.

        With 3 states: diagonal ≈ 0.833, off-diagonal ≈ 0.083.
        This warm-start prevents early EM iterations from collapsing
        states before the model has enough gradient signal.
        """
        raw = self._build_transmat_prior()
        return raw / raw.sum(axis=1, keepdims=True)

    def _label_states(
        self,
        feature_df: pd.DataFrame,
        raw_states: np.ndarray,
    ) -> dict[int, str]:
        """
        Map integer state indices to semantic regime labels.

        Strategy: rank states by their mean daily log-return.
            - Highest mean  → BULL
            - Lowest mean   → BEAR
            - Middle state  → NEUTRAL   (only when n_states == 3)
        For n_states > 3, intermediate states are labelled NEUTRAL_1,
        NEUTRAL_2, … in descending return order.

        Parameters
        ----------
        feature_df : pd.DataFrame
            The original (unscaled) feature DataFrame.  Must contain a
            'Log_Return' column.
        raw_states : np.ndarray, shape (n_obs,)
            Integer state sequence decoded by Viterbi.

        Returns
        -------
        dict[int, str]
            Mapping from HMM state index to semantic label.
        """
        log_ret = feature_df["Log_Return"].values
        state_means: dict[int, float] = {}
        for s in range(self.n_states):
            mask = raw_states == s
            if mask.sum() == 0:
                logger.warning("State %d has zero observations — check n_states.", s)
                state_means[s] = 0.0
            else:
                state_means[s] = float(log_ret[mask].mean())
                logger.info(
                    "State %d: n_obs=%d, mean_log_ret=%.6f",
                    s, mask.sum(), state_means[s],
                )

        # Sort states by mean log-return (ascending: BEAR → BULL)
        sorted_states = sorted(state_means, key=state_means.get)

        mapping: dict[int, str] = {}
        mapping[sorted_states[0]]  = "BEAR"     # lowest return
        mapping[sorted_states[-1]] = "BULL"     # highest return

        middle_states = sorted_states[1:-1]
        if len(middle_states) == 1:
            mapping[middle_states[0]] = "NEUTRAL"
        else:
            for rank, idx in enumerate(reversed(middle_states), start=1):
                mapping[idx] = f"NEUTRAL_{rank}"

        logger.info("State label mapping: %s", mapping)
        return mapping

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        feature_df: pd.DataFrame,
        scaled_matrix: np.ndarray,
    ) -> pd.Series:
        """
        Train the PCA-HMM pipeline and return a labelled regime series.

        Steps
        -----
        1. Project scaled features through PCA (reduces to n_pca_components).
        2. Initialise GaussianHMM with a diagonally dominant transmat_prior.
        3. Train via Baum-Welch (EM algorithm).
        4. Decode the most-likely state path via Viterbi.
        5. Map raw integer states to semantic labels (BULL/BEAR/NEUTRAL).

        Parameters
        ----------
        feature_df : pd.DataFrame
            Clean, aligned feature DataFrame produced by data_engine.
            Index must be a DatetimeIndex.  Must contain 'Log_Return'.
        scaled_matrix : np.ndarray, shape (n_obs, n_features)
            StandardScaler-normalised version of feature_df.values.

        Returns
        -------
        pd.Series
            DatetimeIndex-aligned series of regime labels
            ('BULL', 'BEAR', 'NEUTRAL') with name='Regime'.
        """
        n_obs, n_features = scaled_matrix.shape
        logger.info(
            "Starting PCA-HMM fit: n_obs=%d, n_features=%d, "
            "n_states=%d, n_pca=%d",
            n_obs, n_features, self.n_states, self.n_pca_components,
        )

        # ── Step 1: PCA dimensionality reduction ──────────────────────────
        pca_features = self.pca.fit_transform(scaled_matrix)
        explained = self.pca.explained_variance_ratio_
        logger.info(
            "PCA explained variance: %s  (cumulative: %.2f%%)",
            np.round(explained, 4),
            explained.cumsum()[-1] * 100,
        )

        # ── Step 2: Build GaussianHMM with persistence prior ─────────────
        #   init_params='mc'  → randomly initialise Means and Covariances
        #                        only; startprob_ and transmat_ are set
        #                        manually below for a controlled warm-start.
        #   params='stmc'     → train all four parameter groups during EM.
        self.hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            init_params="mc",    # only random-init means & covariances
            params="stmc",       # train startprob, transmat, means, covars
            transmat_prior=self._build_transmat_prior(),
            verbose=False,
        )

        # Manual initialisation for parameters excluded from init_params
        self.hmm.startprob_ = np.full(
            self.n_states, 1.0 / self.n_states, dtype=float
        )
        self.hmm.transmat_ = self._build_init_transmat()

        # ── Step 3: Baum-Welch EM training ───────────────────────────────
        logger.info("Fitting HMM via Baum-Welch …")
        self.hmm.fit(pca_features)

        n_iters_run = self.hmm.monitor_.iter
        converged    = self.hmm.monitor_.converged
        final_ll     = list(self.hmm.monitor_.history)[-1] if self.hmm.monitor_.history else float("nan")
        logger.info(
            "HMM training complete: iters=%d, converged=%s, final_log_prob=%.4f",
            n_iters_run, converged, final_ll,
        )
        if not converged:
            logger.warning(
                "HMM did not converge within %d iterations. "
                "Consider increasing n_iter or checking data quality.",
                self.n_iter,
            )

        # ── Step 4: Viterbi decoding ──────────────────────────────────────
        raw_states: np.ndarray = self.hmm.predict(pca_features)
        logger.info(
            "Viterbi decoded %d states. State counts: %s",
            len(raw_states),
            {s: int((raw_states == s).sum()) for s in range(self.n_states)},
        )

        # ── Step 5: Semantic labelling ────────────────────────────────────
        self.state_map = self._label_states(feature_df, raw_states)
        self.reverse_map = {v: k for k, v in self.state_map.items()}

        labeled_states = pd.Series(
            [self.state_map[s] for s in raw_states],
            index=feature_df.index,
            name="Regime",
            dtype="category",
        )

        self._is_fitted = True
        logger.info("fit_predict complete. Regime counts:\n%s", labeled_states.value_counts())
        return labeled_states

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Return the learned transition matrix as a labelled DataFrame.

        Rows = 'From' state, Columns = 'To' state.
        Values are probabilities (each row sums to 1).

        Raises
        ------
        RuntimeError
            If called before fit_predict.
        """
        self._check_fitted("get_transition_matrix")
        labels = [self.state_map[s] for s in range(self.n_states)]
        return pd.DataFrame(
            self.hmm.transmat_,
            index=pd.Index(labels, name="From"),
            columns=pd.Index(labels, name="To"),
        ).round(4)

    def get_state_statistics(
        self,
        feature_df: pd.DataFrame,
        labeled_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute per-regime descriptive statistics on log-returns.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Original (unscaled) feature DataFrame with 'Log_Return'.
        labeled_series : pd.Series
            Output of fit_predict — the regime label for each date.

        Returns
        -------
        pd.DataFrame
            Index = regime labels, columns:
                Obs          – number of days assigned to this regime
                Pct_Time     – fraction of total observations (%)
                Mean_Daily   – mean daily log-return
                Ann_Return   – annualised return (Mean_Daily × 252)
                Daily_Vol    – daily std of log-returns
                Ann_Vol      – annualised volatility (Daily_Vol × √252)
                Sharpe       – Ann_Return / Ann_Vol  (risk-free = 0)

        Raises
        ------
        RuntimeError
            If called before fit_predict.
        """
        self._check_fitted("get_state_statistics")

        log_ret = feature_df["Log_Return"]
        records: list[dict] = []

        # Iterate in display order (BULL first)
        present_labels = [
            lbl for lbl in _ORDERED_LABELS if lbl in labeled_series.values
        ] + [
            lbl for lbl in labeled_series.unique() if lbl not in _ORDERED_LABELS
        ]

        for label in present_labels:
            mask = labeled_series == label
            obs = int(mask.sum())
            if obs == 0:
                continue
            ret_slice = log_ret[mask]
            mean_d = float(ret_slice.mean())
            std_d  = float(ret_slice.std(ddof=1)) if obs > 1 else 0.0
            ann_ret = mean_d * _TRADING_DAYS
            ann_vol = std_d  * np.sqrt(_TRADING_DAYS)
            sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

            records.append(
                {
                    "Regime":    label,
                    "Obs":       obs,
                    "Pct_Time":  round(obs / len(labeled_series) * 100, 1),
                    "Mean_Daily":  round(mean_d,  6),
                    "Ann_Return":  round(ann_ret, 4),
                    "Daily_Vol":   round(std_d,   6),
                    "Ann_Vol":     round(ann_vol, 4),
                    "Sharpe":      round(sharpe,  3) if not np.isnan(sharpe) else np.nan,
                }
            )

        stats_df = pd.DataFrame(records).set_index("Regime")
        return stats_df

    def summary(self) -> dict:
        """
        Return a dictionary of key model diagnostics.

        Includes PCA explained variance, HMM convergence info,
        learned transition matrix (raw), and state label mapping.

        Raises
        ------
        RuntimeError
            If called before fit_predict.
        """
        self._check_fitted("summary")
        history = list(self.hmm.monitor_.history)
        return {
            "n_states":                 self.n_states,
            "n_pca_components":         self.n_pca_components,
            "pca_explained_variance":   self.pca.explained_variance_ratio_.tolist(),
            "pca_cumulative_variance":  float(self.pca.explained_variance_ratio_.sum()),
            "hmm_converged":            self.hmm.monitor_.converged,
            "hmm_iterations":           self.hmm.monitor_.iter,
            "hmm_log_prob_history":     history,
            "hmm_final_log_prob":       history[-1] if history else None,
            "state_label_map":          self.state_map,
            "transmat_learned":         self.hmm.transmat_.tolist(),
        }

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _check_fitted(self, caller: str) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{caller}() called before fit_predict(). "
                "Train the model first."
            )


# ---------------------------------------------------------------------------
# Quick self-test (run: python model_engine.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")

    # ── Synthetic data mirroring data_engine output ───────────────────────
    np.random.seed(0)
    N = 600
    dates = pd.bdate_range("2010-01-04", periods=N)

    # Simulate 3 latent regimes in the synthetic log-return series
    returns = np.concatenate([
        np.random.normal(+0.0008, 0.008, 200),   # Bull-like
        np.random.normal(-0.0010, 0.018, 200),   # Bear-like
        np.random.normal(+0.0001, 0.011, 200),   # Neutral-like
    ])
    np.random.shuffle(returns)

    feature_df = pd.DataFrame(
        {
            "Log_Return": returns,
            "Vol_21d":    pd.Series(returns).rolling(21).std().fillna(0.01).values,
            "Vol_63d":    pd.Series(returns).rolling(63).std().fillna(0.01).values,
            "Credit_Stress": np.random.normal(0, 0.01, N),
            "Unemployment_Rate": np.linspace(9.5, 5.0, N),
            "Fed_Funds_Rate":    np.linspace(0.1, 0.5, N),
        },
        index=dates,
    )

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df.values)

    # ── Fit model ─────────────────────────────────────────────────────────
    print("\nRunning model_engine smoke test …\n")
    model = RegimeDetectionModel(n_states=3, n_pca_components=3, n_iter=200)
    regimes = model.fit_predict(feature_df, scaled)

    # ── Results ───────────────────────────────────────────────────────────
    print("=" * 64)
    print("  MODEL SUMMARY")
    print("=" * 64)
    info = model.summary()
    print(f"  PCA explained variance : {np.round(info['pca_explained_variance'], 4)}")
    print(f"  PCA cumulative         : {info['pca_cumulative_variance']:.2%}")
    print(f"  HMM converged          : {info['hmm_converged']}")
    print(f"  HMM iterations         : {info['hmm_iterations']}")
    print(f"  Final log-prob         : {info['hmm_final_log_prob']:.4f}")
    print(f"  State label map        : {info['state_label_map']}")
    print()

    print("  TRANSITION MATRIX (learned)")
    print(model.get_transition_matrix().to_string())
    print()

    print("  REGIME STATISTICS")
    print(model.get_state_statistics(feature_df, regimes).to_string())
    print()

    print("  REGIME SERIES (first 10)")
    print(regimes.head(10).to_string())
    print()

    # ── Assertions ────────────────────────────────────────────────────────
    assert set(regimes.unique()).issubset({"BULL", "BEAR", "NEUTRAL"}), \
        f"Unexpected labels: {set(regimes.unique())}"
    assert len(regimes) == N
    assert regimes.index.equals(dates)
    assert model._is_fitted

    stats = model.get_state_statistics(feature_df, regimes)
    bull_ret  = stats.loc["BULL",  "Mean_Daily"]
    bear_ret  = stats.loc["BEAR",  "Mean_Daily"]
    assert bull_ret > bear_ret, "BULL mean return must exceed BEAR"

    tm = model.get_transition_matrix()
    diag = np.diag(tm.values)
    assert (diag > 0.5).all(), f"Transition matrix diagonal not dominant: {diag}"

    print("=" * 64)
    print("  ALL ASSERTIONS PASSED")
    print("=" * 64)
