"""Run retrievers over the synthetic corpus and expose per-query paired scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .corpus import Corpus
from .metrics import buried, ndcg_at_k, recall_at_k, reciprocal_rank, success_at_1


@dataclass
class RetrievalRun:
    name: str
    per_query: Dict[str, Dict[str, float]]  # metric -> array aligned to corpus.queries
    domains: List[str]

    def metric(self, m: str) -> np.ndarray:
        return self.per_query[m]  # type: ignore[return-value]

    def summary(self) -> Dict[str, float]:
        return {m: float(np.mean(v)) for m, v in self.per_query.items()}

    def by_domain(self, m: str) -> Dict[str, np.ndarray]:
        arr = np.asarray(self.per_query[m])
        doms = np.asarray(self.domains)
        return {d: arr[doms == d] for d in sorted(set(self.domains))}


def evaluate(retriever, corpus: Corpus, k: int = 10) -> RetrievalRun:
    cols: Dict[str, list] = {"success@1": [], f"recall@{k}": [], "mrr": [], f"ndcg@{k}": [], "buried": []}
    for q in corpus.queries:
        ranked = retriever.rank(q.text, k)
        rel = corpus.qrels[q.id]
        cols["success@1"].append(success_at_1(ranked, rel))
        cols[f"recall@{k}"].append(recall_at_k(ranked, rel, k))
        cols["mrr"].append(reciprocal_rank(ranked, rel))
        cols[f"ndcg@{k}"].append(ndcg_at_k(ranked, rel, k))
        cols["buried"].append(buried(ranked, rel, k))
    return RetrievalRun(retriever.name, {m: np.array(v) for m, v in cols.items()}, [q.domain for q in corpus.queries])
