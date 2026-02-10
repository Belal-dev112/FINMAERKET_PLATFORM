"""
Market Forecasting Engine
Implements ARIMA-style, Linear Regression, and Exponential Smoothing models
for short-term price prediction.
"""
import math
import random
from typing import Dict, List, Optional


class LinearTrendModel:
    """Simple linear regression forecaster"""
    def __init__(self):
        self.slope = 0
        self.intercept = 0
        self.fitted = False

    def fit(self, prices: List[float]):
        n = len(prices)
        if n < 2:
            return self
        x_mean = (n - 1) / 2
        y_mean = sum(prices) / n
        num = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        self.slope = num / den if den != 0 else 0
        self.intercept = y_mean - self.slope * x_mean
        self.n = n
        self.fitted = True
        return self

    def predict(self, steps: int) -> List[float]:
        if not self.fitted:
            return []
        return [self.slope * (self.n + i) + self.intercept for i in range(steps)]

    def r_squared(self, prices: List[float]) -> float:
        if not self.fitted:
            return 0
        y_mean = sum(prices) / len(prices)
        ss_res = sum((prices[i] - (self.slope * i + self.intercept)) ** 2 for i in range(len(prices)))
        ss_tot = sum((p - y_mean) ** 2 for p in prices)
        return round(1 - ss_res / ss_tot, 4) if ss_tot != 0 else 0


class ExponentialSmoothingModel:
    """Holt's double exponential smoothing for trend"""
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.level = 0
        self.trend = 0
        self.fitted = False

    def fit(self, prices: List[float]):
        if len(prices) < 3:
            return self
        self.level = prices[0]
        self.trend = prices[1] - prices[0]
        
        for p in prices[1:]:
            prev_level = self.level
            self.level = self.alpha * p + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        
        self.fitted = True
        return self

    def predict(self, steps: int) -> List[float]:
        if not self.fitted:
            return []
        return [self.level + (i + 1) * self.trend for i in range(steps)]


class ARIMALite:
    """Lightweight ARIMA(1,1,1) approximation using rolling statistics"""
    def __init__(self):
        self.ar_coef = 0.5
        self.ma_coef = 0.3
        self.fitted = False
        self.last_diff = 0
        self.last_error = 0
        self.last_price = 0

    def fit(self, prices: List[float]):
        if len(prices) < 5:
            return self
        diffs = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        # Estimate AR coefficient from autocorrelation
        if len(diffs) > 1:
            mean_d = sum(diffs) / len(diffs)
            cov = sum((diffs[i] - mean_d) * (diffs[i-1] - mean_d) for i in range(1, len(diffs)))
            var = sum((d - mean_d) ** 2 for d in diffs)
            self.ar_coef = min(max(cov / var if var != 0 else 0, -0.9), 0.9)
        
        self.last_diff = diffs[-1] if diffs else 0
        self.last_error = 0
        self.last_price = prices[-1]
        self.mean_diff = sum(diffs) / len(diffs) if diffs else 0
        self.fitted = True
        return self

    def predict(self, steps: int) -> List[float]:
        if not self.fitted:
            return []
        preds = []
        d = self.last_diff
        e = self.last_error
        price = self.last_price
        
        for _ in range(steps):
            next_d = self.ar_coef * d + self.ma_coef * e + self.mean_diff
            price = price + next_d
            preds.append(price)
            e = next_d - (self.ar_coef * d + self.mean_diff)
            d = next_d
        return preds


class MarketForecaster:
    def __init__(self):
        self.models = {
            "arima": ARIMALite(),
            "linear": LinearTrendModel(),
            "holt": ExponentialSmoothingModel(),
        }

    def predict(self, prices: List[float], horizon: int = 10) -> Dict:
        if len(prices) < 10:
            return {"direction": "neutral", "prices": [], "confidence": 0}
        
        # Use recent window for fitting
        window = min(len(prices), 100)
        train_prices = prices[-window:]
        
        # Fit all models
        self.models["arima"].fit(train_prices)
        self.models["linear"].fit(train_prices)
        self.models["holt"].fit(train_prices)
        
        arima_preds = self.models["arima"].predict(horizon)
        linear_preds = self.models["linear"].predict(horizon)
        holt_preds = self.models["holt"].predict(horizon)
        
        # Ensemble: weighted average
        if arima_preds and linear_preds and holt_preds:
            ensemble = [
                0.5 * a + 0.3 * h + 0.2 * l
                for a, h, l in zip(arima_preds, holt_preds, linear_preds)
            ]
        else:
            ensemble = arima_preds or linear_preds or holt_preds or [prices[-1]] * horizon
        
        # Add small noise to avoid flat predictions
        ensemble = [max(p + random.gauss(0, prices[-1] * 0.001), prices[-1] * 0.5) for p in ensemble]
        
        current = prices[-1]
        final_pred = ensemble[-1] if ensemble else current
        direction = "up" if final_pred > current * 1.001 else ("down" if final_pred < current * 0.999 else "neutral")
        
        # Confidence: inverse of recent volatility
        recent_returns = [abs((prices[i] - prices[i-1]) / prices[i-1]) for i in range(-20, 0) if prices[i-1] != 0]
        avg_vol = sum(recent_returns) / len(recent_returns) if recent_returns else 0.01
        confidence = round(max(10, min(90, 70 - avg_vol * 1000)), 1)
        
        r2 = self.models["linear"].r_squared(train_prices)
        
        return {
            "direction": direction,
            "prices": [round(p, 2) for p in ensemble],
            "confidence": confidence,
            "r_squared": r2,
            "models": {
                "arima": [round(p, 2) for p in arima_preds[:horizon]],
                "linear": [round(p, 2) for p in linear_preds[:horizon]],
                "holt": [round(p, 2) for p in holt_preds[:horizon]],
            },
            "current_price": round(current, 2),
            "target_price": round(final_pred, 2),
            "expected_change_pct": round((final_pred - current) / current * 100, 3)
        }
