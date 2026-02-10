"""
Authentication & User Management
JWT-based auth with watchlist management
"""
import hashlib
import hmac
import json
import time
import base64
from typing import Dict, List, Optional
from fastapi import HTTPException, Header


SECRET_KEY = "finmarket_secret_key_change_in_production"

# In-memory user store (replace with DB in production)
USERS: Dict[str, Dict] = {
    "demo": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "watchlist": ["AAPL", "MSFT", "BTC-USD"],
        "created_at": time.time()
    }
}


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _create_token(username: str) -> str:
    """Simple JWT-like token (Base64 encoded payload + HMAC)"""
    payload = json.dumps({"username": username, "exp": time.time() + 86400})
    encoded = base64.b64encode(payload.encode()).decode()
    sig = hmac.new(SECRET_KEY.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{sig}"


def _verify_token(token: str) -> Optional[Dict]:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        encoded, sig = parts
        expected_sig = hmac.new(SECRET_KEY.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        payload = json.loads(base64.b64decode(encoded).decode())
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


class AuthManager:
    def register(self, username: str, password: str) -> Dict:
        if not username or not password:
            raise HTTPException(400, "Username and password required")
        if username in USERS:
            raise HTTPException(409, "Username already exists")
        if len(password) < 6:
            raise HTTPException(400, "Password must be at least 6 characters")
        
        USERS[username] = {
            "password_hash": _hash_password(password),
            "watchlist": ["AAPL", "SPY"],
            "created_at": time.time()
        }
        token = _create_token(username)
        return {"token": token, "username": username, "message": "Registration successful"}

    def login(self, username: str, password: str) -> Dict:
        if not username or not password:
            raise HTTPException(400, "Username and password required")
        user = USERS.get(username)
        if not user or user["password_hash"] != _hash_password(password):
            raise HTTPException(401, "Invalid credentials")
        token = _create_token(username)
        return {"token": token, "username": username, "watchlist": user["watchlist"]}

    def get_watchlist(self, username: str) -> Dict:
        user = USERS.get(username)
        if not user:
            raise HTTPException(404, "User not found")
        return {"watchlist": user["watchlist"]}

    def add_to_watchlist(self, username: str, symbol: str) -> Dict:
        user = USERS.get(username)
        if not user:
            raise HTTPException(404, "User not found")
        if symbol not in user["watchlist"]:
            user["watchlist"].append(symbol)
        return {"watchlist": user["watchlist"]}

    def remove_from_watchlist(self, username: str, symbol: str) -> Dict:
        user = USERS.get(username)
        if not user:
            raise HTTPException(404, "User not found")
        user["watchlist"] = [s for s in user["watchlist"] if s != symbol]
        return {"watchlist": user["watchlist"]}


async def get_current_user(authorization: str = Header(None)) -> Dict:
    """FastAPI dependency for authenticated routes"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization[7:]
    payload = _verify_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    return payload
