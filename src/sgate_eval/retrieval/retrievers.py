"""Lightweight retrievers (pure numpy): BM25, TF-IDF cosine, RRF hybrid, authority-aware rerank."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from .corpus import Doc

_TOK = re.compile(r"[a-z0-9\-]+")


def _tok(s: str) -> List[str]:
    return _TOK.findall(s.lower())


class BM25:
    name = "bm25"

    def __init__(self, docs: Sequence[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self.tf = [Counter(_tok(d.text)) for d in self.docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean()
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for t in set(_tok(query)):
            if t not in self.idf:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(t, 0)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out

    def rank(self, query: str, k: int = 10) -> List[str]:
        s = self.scores(query)
        return [self.docs[i].id for i in np.argsort(-s, kind="stable")[:k]]


class TfIdf:
    name = "tfidf"

    def __init__(self, docs: Sequence[Doc]):
        self.docs = list(docs)
        toks = [Counter(_tok(d.text)) for d in self.docs]
        df: Counter = Counter()
        for c in toks:
            df.update(c.keys())
        self.vocab = {t: i for i, t in enumerate(df)}
        n = len(self.docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in self.vocab])
        self.mat = np.stack([self._vec(c) for c in toks])

    def _vec(self, c: Counter) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, f in c.items():
            if t in self.vocab:
                v[self.vocab[t]] = (1 + math.log(f)) * self.idf[self.vocab[t]]
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def scores(self, query: str) -> np.ndarray:
        return self.mat @ self._vec(Counter(_tok(query)))

    def rank(self, query: str, k: int = 10) -> List[str]:
        s = self.scores(query)
        return [self.docs[i].id for i in np.argsort(-s, kind="stable")[:k]]


class RRFHybrid:
    """Reciprocal rank fusion of BM25 and TF-IDF."""
    name = "hybrid_rrf"

    def __init__(self, docs: Sequence[Doc], k: int = 60):
        self.docs = list(docs)
        self.parts = [BM25(docs), TfIdf(docs)]
        self.k = k

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for p in self.parts:
            order = np.argsort(-p.scores(query), kind="stable")
            for rank, i in enumerate(order):
                out[i] += 1.0 / (self.k + rank + 1)
        return out

    def rank(self, query: str, k: int = 10) -> List[str]:
        s = self.scores(query)
        return [self.docs[i].id for i in np.argsort(-s, kind="stable")[:k]]


class AuthorityRerank:
    """Re-ranks a base retriever's candidates using document status metadata.

    final = normalised base score + weight * authority prior, where the prior is
    1 when the recorded (possibly noisy) metadata says authoritative, 0 otherwise. This is the intervention being
    evaluated, not a claim about production systems.
    """
    def __init__(self, base, weight: float = 1.0, depth: int = 20):
        self.base, self.weight, self.depth = base, weight, depth
        self.name = f"{base.name}+authority"
        self.by_id: Dict[str, Doc] = {d.id: d for d in base.docs}

    def rank(self, query: str, k: int = 10) -> List[str]:
        s = self.base.scores(query)
        top = np.argsort(-s, kind="stable")[: self.depth]
        mx = s[top].max() or 1.0
        rescored = [
            (s[i] / mx + self.weight * (1.0 if self.base.docs[i].meta_status == "authoritative" else 0.0), i) for i in top
        ]
        rescored.sort(key=lambda x: -x[0])
        return [self.base.docs[i].id for _, i in rescored[:k]]
