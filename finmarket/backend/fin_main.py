"""
FinMarket Intelligence Platform - Backend API Server
FastAPI-based REST + WebSocket server for real-time market analytics
"""
import asyncio
import webbrowser
import json
import math
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from analytics import FinancialAnalytics
from anomaly import AnomalyDetector
from forecasting import MarketForecaster
from risk import RiskScorer
from database import TimeSeriesDB
from auth import AuthManager, get_current_user

app = FastAPI(
    title="FinMarket Intelligence Platform",
    description="Real-time AI-powered financial market analytics and predictions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Services ---
db = TimeSeriesDB()
analytics = FinancialAnalytics(db)
anomaly_detector = AnomalyDetector()
forecaster = MarketForecaster()
risk_scorer = RiskScorer()
auth_manager = AuthManager()

# In-memory price cache for speed
price_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
connected_clients: List[WebSocket] = []

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BTC-USD", "ETH-USD", "SPY"]

BASE_PRICES = {
    "AAPL": 189.5, "MSFT": 415.2, "GOOGL": 175.8, "AMZN": 198.3,
    "TSLA": 248.7, "NVDA": 875.4, "META": 512.6, "BTC-USD": 67420.0,
    "ETH-USD": 3812.5, "SPY": 523.1
}

# Seed historical data
def seed_historical_data():
    for symbol in SYMBOLS:
        price = BASE_PRICES[symbol]
        now = time.time()
        for i in range(300, 0, -1):
            ts = now - i * 60
            change = random.gauss(0, price * 0.008)
            price = max(price + change, price * 0.5)
            volume = random.randint(100000, 5000000)
            db.insert(symbol, ts, price, volume)
            price_cache[symbol].append({
                "timestamp": ts, "price": price, "volume": volume
            })

seed_historical_data()

# --- Market Data Streaming Simulation ---
async def simulate_market_feed():
    """Simulate real-time market data feed (replace with live API in production)"""
    prices = {s: BASE_PRICES[s] for s in SYMBOLS}
    
    while True:
        tick_data = {}
        for symbol in SYMBOLS:
            # Simulate price movement with drift + volatility
            volatility = 0.003 if "BTC" in symbol or "ETH" in symbol else 0.001
            change_pct = random.gauss(0, volatility)
            # Occasionally inject anomalies
            if random.random() < 0.02:
                change_pct *= random.uniform(3, 8) * random.choice([-1, 1])
            
            prices[symbol] *= (1 + change_pct)
            volume = int(random.lognormvariate(13, 1))
            ts = time.time()
            
            db.insert(symbol, ts, prices[symbol], volume)
            price_cache[symbol].append({
                "timestamp": ts,
                "price": prices[symbol],
                "volume": volume,
                "change_pct": change_pct * 100
            })
            
            tick_data[symbol] = {
                "symbol": symbol,
                "price": round(prices[symbol], 2),
                "volume": volume,
                "change_pct": round(change_pct * 100, 4),
                "timestamp": ts
            }
        
        # Broadcast to WebSocket clients
        if connected_clients:
            msg = json.dumps({"type": "tick", "data": tick_data})
            dead = []
            for ws in connected_clients:
                try:
                    await ws.send_text(msg)
                except:
                    dead.append(ws)
            for ws in dead:
                connected_clients.remove(ws)
        
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup():
    asyncio.create_task(simulate_market_feed())
    # Auto-open the dashboard in the default browser
    webbrowser.open("http://localhost:8000")

# ============================================================
# REST API Endpoints
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "symbols": len(SYMBOLS)}

@app.get("/api/symbols")
def get_symbols():
    return {"symbols": SYMBOLS}

@app.get("/api/price/{symbol}")
def get_current_price(symbol: str):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise HTTPException(404, f"Symbol {symbol} not found")
    cache = list(price_cache[symbol])
    if not cache:
        raise HTTPException(503, "No data available")
    latest = cache[-1]
    prev = cache[-2] if len(cache) > 1 else latest
    return {
        "symbol": symbol,
        "price": round(latest["price"], 2),
        "change": round(latest["price"] - prev["price"], 2),
        "change_pct": round((latest["price"] - prev["price"]) / prev["price"] * 100, 4),
        "volume": latest["volume"],
        "timestamp": latest["timestamp"]
    }

@app.get("/api/history/{symbol}")
def get_history(symbol: str, limit: int = 100):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise HTTPException(404, f"Symbol {symbol} not found")
    data = list(price_cache[symbol])[-limit:]
    return {"symbol": symbol, "data": data, "count": len(data)}

@app.get("/api/indicators/{symbol}")
def get_indicators(symbol: str):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise HTTPException(404, f"Symbol {symbol} not found")
    prices = [d["price"] for d in list(price_cache[symbol])]
    volumes = [d["volume"] for d in list(price_cache[symbol])]
    indicators = analytics.compute_all(prices, volumes)
    return {"symbol": symbol, "indicators": indicators, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/anomalies/{symbol}")
def get_anomalies(symbol: str):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise HTTPException(404, f"Symbol {symbol} not found")
    data = list(price_cache[symbol])
    prices = [d["price"] for d in data]
    volumes = [d["volume"] for d in data]
    anomalies = anomaly_detector.detect(prices, volumes, symbol)
    return {"symbol": symbol, "anomalies": anomalies}

@app.get("/api/forecast/{symbol}")
def get_forecast(symbol: str, horizon: int = 10):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise HTTPException(404, f"Symbol {symbol} not found")
    prices = [d["price"] for d in list(price_cache[symbol])]
    forecast = forecaster.predict(prices, horizon=min(horizon, 30))
    return {"symbol": symbol, "forecast": forecast, "horizon": horizon}

@app.get("/api/risk/{symbol}")
def get_risk(symbol: str):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise HTTPException(404, f"Symbol {symbol} not found")
    data = list(price_cache[symbol])
    prices = [d["price"] for d in data]
    volumes = [d["volume"] for d in data]
    indicators = analytics.compute_all(prices, volumes)
    anomalies = anomaly_detector.detect(prices, volumes, symbol)
    risk = risk_scorer.score(prices, indicators, anomalies)
    return {"symbol": symbol, "risk": risk, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/dashboard")
def get_dashboard():
    """Aggregate dashboard data for all symbols"""
    result = {}
    for symbol in SYMBOLS:
        data = list(price_cache[symbol])
        if len(data) < 10:
            continue
        prices = [d["price"] for d in data]
        volumes = [d["volume"] for d in data]
        latest = data[-1]
        prev = data[-2]
        indicators = analytics.compute_all(prices, volumes)
        anomalies = anomaly_detector.detect(prices, volumes, symbol)
        risk = risk_scorer.score(prices, indicators, anomalies)
        forecast = forecaster.predict(prices, horizon=5)
        
        result[symbol] = {
            "price": round(latest["price"], 2),
            "change_pct": round((latest["price"] - prev["price"]) / prev["price"] * 100, 4),
            "volume": latest["volume"],
            "rsi": indicators.get("rsi"),
            "volatility": indicators.get("volatility"),
            "risk_score": risk["score"],
            "risk_level": risk["level"],
            "anomaly_count": anomalies.get("total_price_anomalies", 0),
            "forecast_direction": forecast.get("direction", "neutral"),
            "forecast_price": forecast.get("prices", [latest["price"]])[-1],
        }
    return result

@app.get("/api/market/overview")
def market_overview():
    """High-level market statistics"""
    gainers, losers = [], []
    total_volume = 0
    for symbol in SYMBOLS:
        data = list(price_cache[symbol])
        if len(data) < 2:
            continue
        change = (data[-1]["price"] - data[0]["price"]) / data[0]["price"] * 100
        total_volume += sum(d["volume"] for d in data[-10:])
        entry = {"symbol": symbol, "change_pct": round(change, 2)}
        (gainers if change > 0 else losers).append(entry)
    
    gainers.sort(key=lambda x: x["change_pct"], reverse=True)
    losers.sort(key=lambda x: x["change_pct"])
    
    return {
        "top_gainers": gainers[:3],
        "top_losers": losers[:3],
        "total_volume_10m": total_volume,
        "active_symbols": len(SYMBOLS),
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================
# Authentication
# ============================================================

@app.post("/api/auth/register")
def register(body: dict):
    return auth_manager.register(body.get("username"), body.get("password"))

@app.post("/api/auth/login")
def login(body: dict):
    return auth_manager.login(body.get("username"), body.get("password"))

@app.get("/api/watchlist")
def get_watchlist(user: dict = Depends(get_current_user)):
    return auth_manager.get_watchlist(user["username"])

@app.post("/api/watchlist/{symbol}")
def add_to_watchlist(symbol: str, user: dict = Depends(get_current_user)):
    return auth_manager.add_to_watchlist(user["username"], symbol.upper())

@app.delete("/api/watchlist/{symbol}")
def remove_from_watchlist(symbol: str, user: dict = Depends(get_current_user)):
    return auth_manager.remove_from_watchlist(user["username"], symbol.upper())

# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws/market")
async def market_websocket(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        # Send initial snapshot
        snapshot = {}
        for symbol in SYMBOLS:
            data = list(price_cache[symbol])
            if data:
                snapshot[symbol] = {"price": round(data[-1]["price"], 2), "volume": data[-1]["volume"]}
        await websocket.send_text(json.dumps({"type": "snapshot", "data": snapshot}))
        
        while True:
            msg = await websocket.receive_text()
            cmd = json.loads(msg)
            if cmd.get("type") == "subscribe":
                await websocket.send_text(json.dumps({"type": "ack", "symbols": cmd.get("symbols", [])}))
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

# --- Serve Frontend Static Files ---
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(str(frontend_dir / "index.html"))
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

if __name__ == "__main__":
    uvicorn.run("fin_main:app", host="0.0.0.0", port=8000, reload=True)
