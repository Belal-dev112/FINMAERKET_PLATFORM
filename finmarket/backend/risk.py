"""
Risk Scoring Engine
Combines volatility, trend strength, anomaly probability, and technical signals
into a comprehensive risk score (0-100)
"""
from typing import Dict, List, Optional


class RiskScorer:
    LEVELS = [
        (20, "MINIMAL", "#00ff88"),
        (40, "LOW", "#88ff00"),
        (60, "MODERATE", "#ffcc00"),
        (75, "HIGH", "#ff8800"),
        (90, "SEVERE", "#ff4400"),
        (101, "CRITICAL", "#ff0000"),
    ]

    def score(self, prices: List[float], indicators: Dict, anomalies: Dict) -> Dict:
        if len(prices) < 5:
            return {"score": 50, "level": "MODERATE", "components": {}}

        components = {}

        # 1. Volatility score (0-30 pts)
        vol = indicators.get("volatility") or 0
        vol_score = min(vol / 2, 30)  # 60% annualized = max 30 pts
        components["volatility"] = round(vol_score, 1)

        # 2. RSI extremes (0-20 pts)
        rsi = indicators.get("rsi") or 50
        rsi_score = 0
        if rsi > 80 or rsi < 20:
            rsi_score = 20
        elif rsi > 70 or rsi < 30:
            rsi_score = 10
        elif rsi > 65 or rsi < 35:
            rsi_score = 5
        components["rsi_risk"] = rsi_score

        # 3. Anomaly score (0-25 pts)
        anomaly_raw = anomalies.get("anomaly_score", 0) if isinstance(anomalies, dict) else 0
        anomaly_score = min(anomaly_raw / 4, 25)
        components["anomaly"] = round(anomaly_score, 1)

        # 4. Momentum reversal (0-15 pts)
        momentum = indicators.get("momentum") or 0
        mom_score = min(abs(momentum) / 2, 15)  # High momentum = more risk
        components["momentum"] = round(mom_score, 1)

        # 5. Price distance from SMA (0-10 pts)
        sma20 = indicators.get("sma_20") or 0
        latest = prices[-1] if prices else 0
        if sma20 > 0:
            dist = abs(latest - sma20) / sma20 * 100
            sma_score = min(dist * 2, 10)
        else:
            sma_score = 0
        components["sma_deviation"] = round(sma_score, 1)

        total = vol_score + rsi_score + anomaly_score + mom_score + sma_score
        total = round(min(max(total, 0), 100), 1)

        # Determine level
        level = "MINIMAL"
        color = "#00ff88"
        for threshold, lvl, clr in self.LEVELS:
            if total < threshold:
                level = lvl
                color = clr
                break

        # Generate recommendations
        recommendations = self._generate_recommendations(total, rsi, vol, indicators, anomalies)

        return {
            "score": total,
            "level": level,
            "color": color,
            "components": components,
            "recommendations": recommendations,
            "summary": self._summary(total, level, components)
        }

    def _generate_recommendations(self, score, rsi, vol, indicators, anomalies):
        recs = []
        if rsi > 75:
            recs.append({"type": "warning", "message": f"RSI at {rsi:.1f} — overbought territory, potential reversal"})
        elif rsi < 25:
            recs.append({"type": "opportunity", "message": f"RSI at {rsi:.1f} — oversold, potential bounce"})
        
        if vol > 40:
            recs.append({"type": "warning", "message": f"High annualized volatility ({vol:.1f}%) — wide price swings expected"})
        
        if isinstance(anomalies, dict) and anomalies.get("anomaly_score", 0) > 50:
            recs.append({"type": "alert", "message": "High anomaly probability — unusual market behavior detected"})
        
        macd = indicators.get("macd")
        if macd and isinstance(macd, dict):
            if macd["histogram"] > 0:
                recs.append({"type": "bullish", "message": "MACD histogram positive — bullish momentum"})
            else:
                recs.append({"type": "bearish", "message": "MACD histogram negative — bearish momentum"})
        
        return recs

    def _summary(self, score, level, components):
        dominant = max(components, key=components.get) if components else "volatility"
        return f"{level} risk ({score:.0f}/100). Primary driver: {dominant.replace('_', ' ').title()}"
