import numpy as np
import pytest

from sgate_eval.gate import Decision, PreRegistration, SliceSpec, run_gate
from sgate_eval.retrieval import (
    BM25, AuthorityRerank, buried, evaluate, generate_corpus, ndcg_at_k, recall_at_k, reciprocal_rank, success_at_1,
)


def test_ndcg_graded_known_value():
    rel = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], rel) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], rel) == pytest.approx(2.8928 / 3.6309, abs=1e-3)
    assert ndcg_at_k(["x", "y"], rel) == 0.0


def test_success_recall_mrr_buried():
    rel = {"a": 2, "b": 1}
    ranked = ["b", "x", "a"]
    assert success_at_1(ranked, rel) == 0.0
    assert recall_at_k(ranked, rel, 3) == 1.0
    assert recall_at_k(ranked, rel, 2) == 0.0
    assert reciprocal_rank(ranked, rel) == pytest.approx(1 / 3)
    assert buried(ranked, rel, 3) == 1.0
    assert buried(["a"], rel, 3) == 0.0


def test_corpus_is_deterministic_and_well_formed():
    c1, c2 = generate_corpus(10, seed=7), generate_corpus(10, seed=7)
    assert [d.id for d in c1.docs] == [d.id for d in c2.docs]
    for q in c1.queries:
        grades = c1.qrels[q.id]
        assert sorted(grades.values()) == [0, 0, 1, 2]


@pytest.fixture(scope="module")
def setup():
    c = generate_corpus(40, seed=1)
    base = evaluate(BM25(c.docs), c)
    rer = evaluate(AuthorityRerank(BM25(c.docs), weight=0.3), c)
    return c, base, rer


def test_plain_lexical_finds_the_right_doc_but_ranks_it_badly(setup):
    _, base, _ = setup
    s = base.summary()
    assert s["recall@10"] > 0.95          # retrieved...
    assert s["success@1"] < 0.6           # ...but often not first
    assert s["buried"] > 0.4


def test_authority_rerank_improves_success_at_1_despite_noisy_metadata(setup):
    _, base, rer = setup
    assert rer.summary()["success@1"] > base.summary()["success@1"] + 0.3
    assert rer.summary()["success@1"] < 1.0  # metadata noise keeps it honest


def test_gate_on_retrieval_runs_end_to_end(setup):
    c, base, rer = setup
    reg = PreRegistration(
        primary_metric="success@1",
        slices=[SliceSpec(name=d, critical=(d == "security"), tolerance=0.02, min_n=30) for d in ["hr", "finance", "security", "product"]],
        min_effect=0.05, n_boot=2000,
    )
    cb, bb = rer.by_domain("success@1"), base.by_domain("success@1")
    rep = run_gate(reg, (rer.metric("success@1"), base.metric("success@1")), {d: (cb[d], bb[d]) for d in cb})
    assert rep.decision == Decision.CANARY
    assert rep.overall["ci_low"] > 0.05
