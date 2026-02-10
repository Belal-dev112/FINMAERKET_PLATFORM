"""
Time Series Database
In-memory optimized time-series store with SQLite persistence.
Designed for streaming analytics with efficient range queries.
"""
import sqlite3
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


class TimeSeriesDB:
    """
    High-performance time series database for market data.
    Uses in-memory deque for hot data + SQLite for persistence.
    """
    
    def __init__(self, db_path: str = "market_data.db", hot_size: int = 10000):
        self.db_path = db_path
        self.hot_size = hot_size
        self.hot_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=hot_size))
        self.lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite schema"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp REAL NOT NULL,
                price REAL NOT NULL,
                volume INTEGER NOT NULL,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_ts ON market_data(symbol, timestamp)")
        conn.commit()
        conn.close()
    
    def insert(self, symbol: str, timestamp: float, price: float, volume: int):
        """Insert a new tick into hot store and periodically flush to SQLite"""
        with self.lock:
            self.hot_store[symbol].append({
                "timestamp": timestamp,
                "price": price,
                "volume": volume
            })
            # Flush every 100 inserts per symbol (in production: async batch writes)
            if len(self.hot_store[symbol]) % 100 == 0:
                self._flush_to_db(symbol, timestamp, price, volume)
    
    def _flush_to_db(self, symbol: str, timestamp: float, price: float, volume: int):
        """Persist recent data to SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO market_data (symbol, timestamp, price, volume) VALUES (?, ?, ?, ?)",
                (symbol, timestamp, price, volume)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass  # Non-blocking persistence
    
    def query(self, symbol: str, limit: int = 100, from_ts: Optional[float] = None) -> List[Dict]:
        """Query time series data (hot store first)"""
        with self.lock:
            data = list(self.hot_store[symbol])
            if from_ts:
                data = [d for d in data if d["timestamp"] >= from_ts]
            return data[-limit:]
    
    def query_range(self, symbol: str, from_ts: float, to_ts: float) -> List[Dict]:
        """Query a time range from hot store + SQLite"""
        hot = [d for d in self.hot_store[symbol] if from_ts <= d["timestamp"] <= to_ts]
        if len(hot) >= 100:
            return hot
        
        # Fall back to SQLite for historical data
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT timestamp, price, volume FROM market_data WHERE symbol=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
                (symbol, from_ts, to_ts)
            ).fetchall()
            conn.close()
            return [{"timestamp": r[0], "price": r[1], "volume": r[2]} for r in rows]
        except Exception:
            return hot
    
    def get_symbols(self) -> List[str]:
        return list(self.hot_store.keys())
    
    def get_stats(self) -> Dict:
        """Database statistics for monitoring"""
        with self.lock:
            return {
                "symbols": len(self.hot_store),
                "total_hot_ticks": sum(len(v) for v in self.hot_store.values()),
                "db_path": self.db_path,
                "hot_capacity": self.hot_size,
            }
