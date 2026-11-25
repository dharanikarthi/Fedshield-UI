"""Demo role-based access. Five personas, HMAC-signed session tokens, and scope
enforced on the SERVER (a Bank-A token asking for a Bank-B account gets 403).

Not production auth: passwords are demo values from FEDSHIELD_DEMO_PASSWORD
(default "fedshield"), the signing key is per-process unless FEDSHIELD_SECRET is set.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from fastapi import Header, HTTPException, Query

SECRET = os.environ.get("FEDSHIELD_SECRET", "").encode() or secrets.token_bytes(32)
PASSWORD = os.environ.get("FEDSHIELD_DEMO_PASSWORD", "fedshield")
TTL = 8 * 3600

USERS = {
    "bank_a": {"role": "bank", "bank": "Bank-A", "name": "Bank A · Fraud officer"},
    "bank_b": {"role": "bank", "bank": "Bank-B", "name": "Bank B · Fraud officer"},
    "bank_c": {"role": "bank", "bank": "Bank-C", "name": "Bank C · Fraud officer"},
    "admin": {"role": "admin", "bank": None, "name": "FedShield aggregation server"},
    "rbi": {"role": "rbi", "bank": None, "name": "RBI supervisory desk"},
}

# what each role can and cannot see — shown verbatim in the UI
SCOPE = {
    "bank": {"can": ["Your bank's accounts, scores and transactions", "Explanations for your alerts",
                     "Hashed consortium registry hits (count only, never who)", "Your own audit trail"],
             "cannot": ["Any other bank's accounts or transactions", "Other banks' labels or alerts",
                        "Individual banks' model updates"]},
    "admin": {"can": ["Which banks are connected and their row counts", "Masked histogram sums per round",
                      "Gate decisions, privacy budget, global model quality"],
              "cannot": ["Any account, transaction or label", "Any single bank's unmasked histogram"]},
    "rbi": {"can": ["Cross-bank clusters via hashed identifiers", "Alert counts, evidence sources, audit trail",
                    "Pattern library and fairness metrics"],
            "cannot": ["Raw customer data, balances or transactions", "Account identities (hashed only)"]},
}


def _b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue(username, password):
    u = USERS.get(username)
    if not u or not hmac.compare_digest(password, PASSWORD):
        raise HTTPException(401, "invalid username or password")
    body = _b64(json.dumps({"sub": username, **u, "exp": int(time.time()) + TTL}).encode())
    sig = _b64(hmac.new(SECRET, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}", {"user": username, **u, "scope": SCOPE[u["role"]]}


def verify(token):
    try:
        body, sig = token.split(".")
        good = _b64(hmac.new(SECRET, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            raise ValueError
        s = json.loads(_unb64(body))
        if s["exp"] < time.time():
            raise ValueError
        return s
    except Exception:
        raise HTTPException(401, "invalid or expired session")


def session(authorization: str = Header(None), token: str = Query(None)):
    raw = (authorization or "").removeprefix("Bearer ").strip() or token
    if not raw:
        raise HTTPException(401, "login required")
    return verify(raw)


def need(s, *roles):
    if s["role"] not in roles:
        raise HTTPException(403, f"role '{s['role']}' cannot access this resource")


def mask_id(account_id):
    """Stable pseudonym for the supervisory view: RBI sees links, never identities."""
    return "H-" + hmac.new(SECRET, account_id.encode(), hashlib.sha256).hexdigest()[:10]
