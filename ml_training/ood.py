"""Out-of-distribution screening for the mock model.

Two complementary signals, because each misses a different kind of stranger:

* Isolation Forest catches points in sparse regions of the training manifold.
* Mahalanobis distance catches extrapolation — values far outside the training spread,
  where trees stop discriminating and the classifier's probability means little.

Both thresholds are fixed on training data at the same nominal flag rate, so the
expected in-distribution false-flag rate is known before anything is served. A flag
says "unfamiliar input", never "bad risk"; in the demo it simply blocks the
straight-through path and routes the case to the reasoning and human-review route.
"""
import numpy as np
from sklearn.ensemble import IsolationForest


class NoveltyDetector:
    def __init__(self, flag_rate=0.01, seed=7, n_estimators=250, distance_columns=None):
        self.flag_rate = flag_rate
        self.seed = seed
        self.n_estimators = n_estimators
        # Distance is measured on the continuous block only. One-hot and missing-indicator
        # columns are near-constant in places, and inverting their covariance turns ordinary
        # categories into false strangers.
        self.distance_columns = distance_columns

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.forest_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed).fit(X)
        block = self._distance_block(X)
        self.mean_ = block.mean(axis=0)
        covariance = np.atleast_2d(np.cov(block, rowvar=False)) + np.eye(block.shape[1]) * 1e-6
        self.inverse_covariance_ = np.linalg.pinv(covariance)
        self.isolation_threshold_ = float(np.quantile(self.forest_.score_samples(X), self.flag_rate))
        self.distance_threshold_ = float(np.quantile(self.mahalanobis(X), 1 - self.flag_rate))
        return self

    def _distance_block(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, :self.distance_columns] if self.distance_columns else X

    def mahalanobis(self, X):
        delta = self._distance_block(X) - self.mean_
        return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', delta, self.inverse_covariance_, delta), 0))

    def scores(self, X):
        return {'isolation': self.forest_.score_samples(np.asarray(X, dtype=float)),
                'mahalanobis': self.mahalanobis(X)}

    def flags(self, X):
        s = self.scores(X)
        return (s['isolation'] < self.isolation_threshold_) | (s['mahalanobis'] > self.distance_threshold_)

    def explain(self, X):
        """Per-row reason, for the audit trail rather than for the applicant."""
        s = self.scores(X)
        out = []
        for isolation, distance in zip(s['isolation'], s['mahalanobis']):
            reasons = []
            if isolation < self.isolation_threshold_:
                reasons.append('sparse region of the training data')
            if distance > self.distance_threshold_:
                reasons.append('values beyond the trained range')
            out.append({'ood': bool(reasons), 'reasons': reasons,
                        'isolation_score': round(float(isolation), 4),
                        'mahalanobis': round(float(distance), 3)})
        return out
