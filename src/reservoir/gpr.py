from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .data import make_lag_features


@dataclass
class GPRReport:
    rmse: float
    r2: float
    mae: float
    qr20: float
    selected_lags: list[int]


class RunoffGPRForecaster:
    def __init__(self, candidate_lags: list[int] | None = None, kernel_type: str = "se"):
        self.candidate_lags = candidate_lags or [1, 2, 3, 6, 12, 24, 36, 48]
        self.selected_lags: list[int] = []
        self.model: GaussianProcessRegressor | None = None
        self.kernel_type = kernel_type

    def _kernel(self):
        if self.kernel_type == "matern":
            base = Matern(length_scale=2.0, nu=1.5)
        elif self.kernel_type == "rq":
            base = RationalQuadratic(length_scale=1.0, alpha=1.0)
        else:
            base = RBF(length_scale=2.0)
        return ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-3, (1e-6, 1e1))

    def select_features(self, qin: np.ndarray, max_features: int = 5) -> list[int]:
        selected: list[int] = []
        remain = list(self.candidate_lags)
        best_score = -np.inf
        for _ in range(min(max_features, len(remain))):
            best_lag, round_best = None, best_score
            for lag in remain:
                trial = sorted(selected + [lag])
                x, y = make_lag_features(qin, trial)
                model = LinearRegression().fit(x, y)
                score = model.score(x, y)
                if score > round_best + 1e-4:
                    round_best, best_lag = score, lag
            if best_lag is None:
                break
            selected.append(best_lag)
            remain.remove(best_lag)
            best_score = round_best
        self.selected_lags = sorted(selected)
        return self.selected_lags

    def fit(self, qin: np.ndarray) -> None:
        if not self.selected_lags:
            self.select_features(qin)
        x, y = make_lag_features(qin, self.selected_lags)
        self.model = GaussianProcessRegressor(kernel=self._kernel(), n_restarts_optimizer=3, normalize_y=True)
        self.model.fit(x, y)

    def predict_next(self, history: np.ndarray) -> float:
        assert self.model is not None, "fit first"
        x = np.array([[history[-l] for l in self.selected_lags]])
        mean, _ = self.model.predict(x, return_std=True)
        return float(max(0.0, mean[0]))

    def rolling_predict(self, history: np.ndarray, steps: int) -> np.ndarray:
        seq = history.copy().tolist()
        out = []
        for _ in range(steps):
            y = self.predict_next(np.array(seq))
            out.append(y)
            seq.append(y)
        return np.asarray(out)

    def evaluate(self, qin: np.ndarray, split: float = 0.8) -> GPRReport:
        cut = int(len(qin) * split)
        train, test = qin[:cut], qin[cut - max(self.candidate_lags):]
        self.fit(train)
        max_lag = max(self.selected_lags)
        preds, obs = [], []
        for i in range(max_lag, len(test)):
            hist = test[:i]
            preds.append(self.predict_next(hist))
            obs.append(test[i])
        pred, obs = np.asarray(preds), np.asarray(obs)
        rmse = mean_squared_error(obs, pred, squared=False)
        mae = float(np.mean(np.abs(obs - pred)))
        r2 = r2_score(obs, pred)
        qr20 = float(np.mean(np.abs(obs - pred) <= 0.2 * np.maximum(obs, 1.0)))
        return GPRReport(rmse=float(rmse), r2=float(r2), mae=mae, qr20=qr20, selected_lags=self.selected_lags)
