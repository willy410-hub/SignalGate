"""Leakage / contamination checks between an eval set and training text.

Three complementary detectors:
- n-gram overlap (default 13 word-grams): catches verbatim and near-verbatim copies.
- MinHash Jaccard estimate: catches near-duplicates with light edits.
- canary strings: a unique token planted in the eval set; finding it in training
  data proves leakage, but its absence proves nothing (it only detects).
Paraphrases evade all three; semantic (embedding) search is a separate layer.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

import numpy as np

_TOKEN = re.compile(r"\w+")
_P = (1 << 61) - 1  # Mersenne prime for universal hashing


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> Set[tuple]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(item: str, corpus_ngrams: Set[tuple], n: int = 13) -> float:
    """Fraction of the item's n-grams that appear anywhere in the corpus."""
    grams = ngrams(tokenize(item), n)
    if not grams:
        return 0.0
    return len(grams & corpus_ngrams) / len(grams)


def build_corpus_ngrams(docs: Iterable[str], n: int = 13) -> Set[tuple]:
    out: Set[tuple] = set()
    for d in docs:
        out |= ngrams(tokenize(d), n)
    return out


def _h(token_tuple: tuple) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(token_tuple).encode(), digest_size=8).digest(), "big")


class MinHasher:
    """MinHash signatures over word shingles; estimates Jaccard similarity."""

    def __init__(self, num_perm: int = 128, shingle: int = 3, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.num_perm, self.shingle = num_perm, shingle
        self.a = rng.integers(1, _P, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _P, size=num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        shingles = ngrams(tokenize(text), self.shingle) or {(t,) for t in tokenize(text)}
        if not shingles:
            return np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        base = np.array([_h(s) % _P for s in shingles], dtype=object)
        sig = np.empty(self.num_perm, dtype=np.uint64)
        for i in range(self.num_perm):
            a, b = int(self.a[i]), int(self.b[i])
            sig[i] = min((a * int(x) + b) % _P for x in base)
        return sig

    @staticmethod
    def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))


@dataclass(frozen=True)
class ContaminationResult:
    index: int
    ngram_overlap: float
    max_minhash: float
    canary_hit: bool
    flagged: bool


def contamination_report(
    eval_items: Sequence[str],
    train_docs: Sequence[str],
    n: int = 13,
    overlap_threshold: float = 0.5,
    minhash_threshold: float = 0.8,
    canaries: Sequence[str] = (),
    hasher: MinHasher | None = None,
) -> List[ContaminationResult]:
    hasher = hasher or MinHasher()
    corpus = build_corpus_ngrams(train_docs, n)
    train_sigs = [hasher.signature(d) for d in train_docs]
    joined = "\n".join(train_docs)
    out: List[ContaminationResult] = []
    for i, item in enumerate(eval_items):
        ov = ngram_overlap(item, corpus, n)
        sig = hasher.signature(item)
        mh = max((hasher.jaccard(sig, s) for s in train_sigs), default=0.0)
        canary = any(c in joined and c in item for c in canaries)
        out.append(
            ContaminationResult(i, ov, mh, canary, bool(ov >= overlap_threshold or mh >= minhash_threshold or canary))
        )
    return out
