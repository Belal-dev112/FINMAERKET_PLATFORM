"""
Anomaly Detection Engine
Detects price spikes, crashes, volume anomalies using Z-score and IQR methods
"""
import math
import statistics
from datetime import datetime
from typing import Dict, List


class AnomalyDetector:
    def __init__(self, z_threshold: float = 3.0, volume_threshold: float = 2.5):
        self.z_threshold = z_threshold
        self.volume_threshold = volume_threshold
        self.anomaly_history: Dict[str, List] = {}

    def z_score_anomaly(self, values: List[float], window: int = 30) -> List[Dict]:
        """Detect anomalies using rolling Z-score"""
        if len(values) < window + 1:
            return []
        
        anomalies = []
        for i in range(window, len(values)):
            window_data = values[i-window:i]
            mean = sum(window_data) / len(window_data)
            variance = sum((x - mean) ** 2 for x in window_data) / len(window_data)
            std = math.sqrt(variance) if variance > 0 else 1e-10
            z = (values[i] - mean) / std
            
            if abs(z) > self.z_threshold:
                anomalies.append({
                    "index": i,
                    "value": round(values[i], 4),
                    "z_score": round(z, 4),
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "type": "high" if z > 0 else "low"
                })
        return anomalies

    def iqr_anomaly(self, values: List[float]) -> List[Dict]:
        """Detect anomalies using IQR method"""
        if len(values) < 10:
            return []
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        anomalies = []
        for i, v in enumerate(values):
            if v < lower or v > upper:
                anomalies.append({
                    "index": i,
                    "value": round(v, 4),
                    "bound": "upper" if v > upper else "lower",
                    "deviation": round(abs(v - (upper if v > upper else lower)), 4)
                })
        return anomalies

    def detect_price_spikes(self, prices: List[float]) -> List[Dict]:
        """Detect sudden price movements"""
        if len(prices) < 2:
            return []
        
        spikes = []
        for i in range(1, len(prices)):
            change = abs(prices[i] - prices[i-1]) / max(prices[i-1], 1e-10)
            if change > 0.05:  # >5% move in one tick
                spikes.append({
                    "index": i,
                    "severity": "critical" if change > 0.1 else "warning",
                    "change_pct": round(change * 100, 3),
                    "direction": "up" if prices[i] > prices[i-1] else "down",
                    "price": round(prices[i], 4)
                })
        return spikes

    def detect_volume_anomalies(self, volumes: List[float]) -> List[Dict]:
        """Detect abnormal trading volume"""
        if len(volumes) < 20:
            return []
        
        anomalies = []
        window = 20
        for i in range(window, len(volumes)):
            avg_vol = sum(volumes[i-window:i]) / window
            if avg_vol == 0:
                continue
            ratio = volumes[i] / avg_vol
            if ratio > 3.0 or ratio < 0.2:
                anomalies.append({
                    "index": i,
                    "volume": int(volumes[i]),
                    "avg_volume": int(avg_vol),
                    "ratio": round(ratio, 2),
                    "type": "volume_surge" if ratio > 3.0 else "volume_drought"
                })
        return anomalies

    def compute_anomaly_score(self, price_anomalies: List, volume_anomalies: List, spikes: List) -> float:
        """0-100 anomaly probability score"""
        score = 0
        # Weight recent anomalies more
        if spikes:
            score += min(len(spikes) * 15, 40)
            if any(s["severity"] == "critical" for s in spikes):
                score += 20
        if price_anomalies:
            score += min(len(price_anomalies) * 5, 25)
        if volume_anomalies:
            score += min(len(volume_anomalies) * 5, 15)
        return round(min(score, 100), 1)

    def detect(self, prices: List[float], volumes: List[float], symbol: str = "") -> List[Dict]:
        """Run all anomaly detection and return summary"""
        price_anomalies = self.z_score_anomaly(prices)
        volume_anomalies = self.detect_volume_anomalies(volumes)
        spikes = self.detect_price_spikes(prices[-50:])
        iqr_anomalies = self.iqr_anomaly(prices[-100:])
        
        # Keep only recent (last 5)
        recent_spikes = spikes[-5:]
        score = self.compute_anomaly_score(price_anomalies[-10:], volume_anomalies[-5:], recent_spikes)
        
        alerts = []
        for spike in recent_spikes:
            alerts.append({
                "type": "price_spike",
                "severity": spike["severity"],
                "message": f"Price {spike['direction']} {spike['change_pct']}% spike detected",
                "timestamp": datetime.utcnow().isoformat()
            })
        for va in volume_anomalies[-3:]:
            alerts.append({
                "type": "volume_anomaly",
                "severity": "warning",
                "message": f"Volume {va['ratio']}x {'surge' if va['ratio'] > 3 else 'drop'} detected",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {
            "anomaly_score": score,
            "total_price_anomalies": len(price_anomalies),
            "total_volume_anomalies": len(volume_anomalies),
            "recent_spikes": recent_spikes,
            "alerts": alerts
        }
