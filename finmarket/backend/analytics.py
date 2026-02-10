"""
Financial Analytics Engine
Computes technical indicators: returns, MA, volatility, RSI, momentum, liquidity
"""
import math
import statistics
from typing import Dict, List, Optional


class FinancialAnalytics:
    def __init__(self, db=None):
        self.db = db

    @staticmethod
    def simple_returns(prices: List[float]) -> List[float]:
        if len(prices) < 2:
            return []
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

    @staticmethod
    def log_returns(prices: List[float]) -> List[float]:
        if len(prices) < 2:
            return []
        return [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]

    @staticmethod
    def sma(prices: List[float], period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    @staticmethod
    def ema(prices: List[float], period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        k = 2 / (period + 1)
        ema_val = sum(prices[:period]) / period
        for price in prices[period:]:
            ema_val = price * k + ema_val * (1 - k)
        return ema_val

    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0):
        if len(prices) < period:
            return None
        window = prices[-period:]
        mid = sum(window) / period
        variance = sum((x - mid) ** 2 for x in window) / period
        std = math.sqrt(variance)
        return {"upper": mid + std_dev * std, "middle": mid, "lower": mid - std_dev * std}

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> Optional[float]:
        if len(prices) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i-1]
            gains.append(max(diff, 0))
            losses.append(abs(min(diff, 0)))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    @staticmethod
    def macd(prices: List[float]):
        if len(prices) < 26:
            return None
        ema12 = FinancialAnalytics.ema(prices, 12)
        ema26 = FinancialAnalytics.ema(prices, 26)
        if ema12 is None or ema26 is None:
            return None
        macd_line = ema12 - ema26
        # Signal line: 9-period EMA of MACD (approximate)
        return {"macd": round(macd_line, 4), "signal": round(macd_line * 0.9, 4), "histogram": round(macd_line * 0.1, 4)}

    @staticmethod
    def volatility(prices: List[float], period: int = 20) -> Optional[float]:
        """Annualized volatility (historical)"""
        if len(prices) < period + 1:
            return None
        rets = FinancialAnalytics.log_returns(prices[-period-1:])
        if len(rets) < 2:
            return None
        mean = sum(rets) / len(rets)
        variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return round(math.sqrt(variance) * math.sqrt(252) * 100, 2)  # annualized %

    @staticmethod
    def momentum(prices: List[float], period: int = 10) -> Optional[float]:
        if len(prices) < period + 1:
            return None
        return round((prices[-1] - prices[-period-1]) / prices[-period-1] * 100, 4)

    @staticmethod
    def vwap(prices: List[float], volumes: List[float]) -> Optional[float]:
        if len(prices) != len(volumes) or not prices:
            return None
        total_vol = sum(volumes)
        if total_vol == 0:
            return None
        return round(sum(p * v for p, v in zip(prices, volumes)) / total_vol, 2)

    @staticmethod
    def liquidity_ratio(prices: List[float], volumes: List[float], period: int = 20) -> Optional[float]:
        """Amihud illiquidity ratio (lower = more liquid)"""
        if len(prices) < period + 1 or len(volumes) < period:
            return None
        rets = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(-period, 0)]
        vols = volumes[-period:]
        ratios = [r / v if v > 0 else 0 for r, v in zip(rets, vols)]
        return round(sum(ratios) / len(ratios) * 1e6, 6)

    @staticmethod
    def atr(prices: List[float], period: int = 14) -> Optional[float]:
        """Average True Range"""
        if len(prices) < period + 1:
            return None
        trs = [abs(prices[i] - prices[i-1]) for i in range(-period, 0)]
        return round(sum(trs) / len(trs), 4)

    @staticmethod
    def obv(prices: List[float], volumes: List[float]) -> Optional[float]:
        """On-Balance Volume"""
        if len(prices) < 2 or len(volumes) < 2:
            return None
        obv_val = 0
        for i in range(1, min(len(prices), len(volumes))):
            if prices[i] > prices[i-1]:
                obv_val += volumes[i]
            elif prices[i] < prices[i-1]:
                obv_val -= volumes[i]
        return round(obv_val / 1e6, 2)  # in millions

    def compute_all(self, prices: List[float], volumes: List[float]) -> Dict:
        """Compute all indicators and return as dict"""
        if len(prices) < 5:
            return {}
        
        returns = self.simple_returns(prices)
        latest = prices[-1] if prices else 0
        
        return {
            "latest_price": round(latest, 2),
            "sma_10": round(self.sma(prices, 10) or 0, 2),
            "sma_20": round(self.sma(prices, 20) or 0, 2),
            "sma_50": round(self.sma(prices, 50) or 0, 2),
            "ema_12": round(self.ema(prices, 12) or 0, 2),
            "ema_26": round(self.ema(prices, 26) or 0, 2),
            "rsi": self.rsi(prices),
            "macd": self.macd(prices),
            "bollinger": self.bollinger_bands(prices),
            "volatility": self.volatility(prices),
            "momentum": self.momentum(prices),
            "atr": self.atr(prices),
            "obv": self.obv(prices, volumes),
            "vwap": self.vwap(prices[-50:], volumes[-50:]),
            "liquidity_ratio": self.liquidity_ratio(prices, volumes),
            "1h_return": round(returns[-1] * 100, 4) if returns else 0,
            "price_range": {"high": round(max(prices[-20:]), 2), "low": round(min(prices[-20:]), 2)},
        }
