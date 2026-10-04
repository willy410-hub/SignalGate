"""SYNTHETIC authority-aware retrieval corpus.

Each topic has an authoritative document (the current official answer), a
superseded older version that says something different, and forum-style
distractors that repeat the old value. Queries ask for the current value.
Everything is generated; no real client or company data is used. The corpus is
designed to reproduce the failure shape 'the right document is retrieved but
not ranked first', so results measure ranking under authority, not real-world
retrieval quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

DOMAINS = {
    "hr": ["parental leave", "remote work stipend", "probation period", "overtime cap"],
    "finance": ["expense limit", "refund window", "invoice approval threshold", "travel per diem"],
    "security": ["password rotation", "session timeout", "vpn retention", "key escrow period"],
    "product": ["trial length", "seat limit", "export quota", "api rate limit"],
}
UNITS = {"hr": "days", "finance": "days", "security": "days", "product": "units"}
SYLLABLES = ["bel", "dar", "kon", "tri", "vex", "lum", "ori", "san", "pex", "zul", "mar", "nod"]

GRADE = {"authoritative": 2, "superseded": 1, "distractor": 0}


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    status: str  # authoritative | superseded | distractor
    topic: str
    domain: str
    meta_status: str = ""  # status label as recorded in (imperfect) metadata


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    topic: str
    domain: str


@dataclass
class Corpus:
    docs: List[Doc]
    queries: List[Query]
    qrels: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _name(rng: np.random.Generator) -> str:
    return "".join(rng.choice(SYLLABLES, size=3)).capitalize()


def generate_corpus(
    n_topics_per_domain: int = 50, seed: int = 0, lexical_leak: float = 0.3, metadata_reliability: float = 0.9
) -> Corpus:
    """`lexical_leak`: chance the authoritative doc literally contains the word 'current'
    (an easy cue). `metadata_reliability`: chance a document's recorded status label is
    correct; real metadata is imperfect, so rerankers that trust it are not given an oracle."""
    rng = np.random.default_rng(seed)
    docs: List[Doc] = []
    queries: List[Query] = []
    qrels: Dict[str, Dict[str, int]] = {}
    t = 0
    for domain, subjects in DOMAINS.items():
        for _ in range(n_topics_per_domain):
            subject = str(rng.choice(subjects))
            entity = _name(rng) + f"-{t}"
            topic = f"{entity}:{subject}"
            new, old = int(rng.integers(5, 60)), int(rng.integers(5, 60))
            if old == new:
                old += 1
            unit = UNITS[domain]
            phrase = f"{subject} for {entity}"
            cue = "current " if rng.random() < lexical_leak else ""

            def meta(true_status: str) -> str:
                if rng.random() < metadata_reliability:
                    return true_status
                return str(rng.choice([x for x in GRADE if x != true_status]))

            def pad(k: int) -> str:  # variable verbosity: repeated mentions of the query phrase
                return " ".join(f"Note: the {phrase} matters for audits." for _ in range(k))

            a_text = f"Official policy v3. The {cue}{phrase} is {new} {unit}. {pad(int(rng.integers(0, 4)))} Approved by the policy board."
            s_text = f"Policy v2 (replaced). The {phrase} is {old} {unit}. {pad(int(rng.integers(0, 4)))} See the {entity} handbook."
            auth = Doc(f"{t}-a", a_text, "authoritative", topic, domain, meta("authoritative"))
            sup = Doc(f"{t}-s", s_text, "superseded", topic, domain, meta("superseded"))
            forum = [
                Doc(
                    f"{t}-f{j}",
                    f"Forum thread: does anyone know the {phrase}? I think it is {old} {unit}. {pad(int(rng.integers(0, 3)))}",
                    "distractor", topic, domain, meta("distractor"),
                )
                for j in range(2)
            ]
            docs += [auth, sup, *forum]
            q = Query(f"q{t}", f"What is the current {phrase}?", topic, domain)
            queries.append(q)
            qrels[q.id] = {auth.id: 2, sup.id: 1, **{f.id: 0 for f in forum}}
            t += 1
    order = rng.permutation(len(docs))
    return Corpus([docs[i] for i in order], queries, qrels)
