"""SQLite results registry and a held-out regression suite.

The registry records every evaluation run with its pre-registration hash so a result
can always be traced to the rules it was judged under. The regression suite stores
items a model once failed; a candidate must not re-fail items that were fixed.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


class Registry:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs(
              id INTEGER PRIMARY KEY, ts REAL, name TEXT, registration_hash TEXT,
              seed INTEGER, decision TEXT, metrics TEXT);
            CREATE TABLE IF NOT EXISTS regression_items(
              item_id TEXT PRIMARY KEY, added_ts REAL, status TEXT, note TEXT);
            """
        )

    def log_run(self, name: str, registration_hash: str, seed: int, decision: str, metrics: Dict[str, float]) -> int:
        cur = self.db.execute(
            "INSERT INTO runs(ts,name,registration_hash,seed,decision,metrics) VALUES(?,?,?,?,?,?)",
            (time.time(), name, registration_hash, seed, decision, json.dumps(metrics)),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def runs(self, name: Optional[str] = None) -> List[dict]:
        q = "SELECT id,name,registration_hash,seed,decision,metrics FROM runs" + (" WHERE name=?" if name else "") + " ORDER BY id"
        rows = self.db.execute(q, (name,) if name else ()).fetchall()
        return [dict(id=r[0], name=r[1], registration_hash=r[2], seed=r[3], decision=r[4], metrics=json.loads(r[5])) for r in rows]

    # ---- regression suite --------------------------------------------------
    def add_failures(self, item_ids: Iterable[str], note: str = "") -> None:
        for i in item_ids:
            self.db.execute("INSERT OR IGNORE INTO regression_items VALUES(?,?,?,?)", (i, time.time(), "open", note))
        self.db.commit()

    def mark_fixed(self, item_ids: Iterable[str]) -> None:
        for i in item_ids:
            self.db.execute("UPDATE regression_items SET status='fixed' WHERE item_id=?", (i,))
        self.db.commit()

    def check_regressions(self, failed_now: Sequence[str]) -> Dict[str, List[str]]:
        """Compare the candidate's failures with the suite. 'regressed' = fixed before, failing again."""
        failed = set(failed_now)
        rows = dict(self.db.execute("SELECT item_id,status FROM regression_items").fetchall())
        return {
            "regressed": sorted(i for i, s in rows.items() if s == "fixed" and i in failed),
            "still_open": sorted(i for i, s in rows.items() if s == "open" and i in failed),
            "newly_fixed": sorted(i for i, s in rows.items() if s == "open" and i not in failed),
        }
