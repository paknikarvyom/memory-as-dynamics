#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory as Dynamics — a web-page memory encoder   (v2)
======================================================

Implements the recurrent attractor + sequence memory model described in
"Memory as Dynamics: A Unified Model of Distributed Sequence Encoding in
Recurrent Neural Systems" (Paknikar, 2026), and wraps it in a small tool that:

  1. scrapes a URL for its readable text,
  2. turns that text into a JSON record of ordered "chunks",
  3. encodes each chunk as a distributed bipolar pattern and stores it in a
     recurrent network as an attractor memory (Hebbian rule) and as a step in
     a sequence memory (temporal / successor-predecessor rule),
  4. visualises recall, sequence replay and basic capacity diagnostics in a
     QtPy desktop UI.

Paper -> code map
------------------
  Eq. (1), (11)   DynamicMemoryNetwork.step         x(t+1)=tanh(beta*(W_A x + g(t) W_T x + b))
  Eq. (3), (4)    DynamicMemoryNetwork.spectral_radius_at   stability check, rho(J) < 1
  Eq. (5), (8)    DynamicMemoryNetwork.store_patterns       Hebbian attractor matrix W_A
  Eq. (6), (9)    DynamicMemoryNetwork.store_sequence       temporal / successor-predecessor matrix W_T
  Eq. (7)         gamma decay applied to W_A / W_T (homeostatic regularization)
  Eq. (12)        DynamicMemoryNetwork.gate                theta-like periodic gate g(t)
  Eq. (14), (15)  DynamicMemoryNetwork.capacity_estimates   Pmax ~ 0.14N, Smax ~ N/(c ln N)
  Eq. (16)        DynamicMemoryNetwork.energy_proxy         event-based energy proxy
  Table 2 (m_mu)  DynamicMemoryNetwork.overlap              pattern/state overlap

Changes in v2 (everything marked EXTENSION departs from the paper's equations
and is OFF by default in the library API, so results stay paper-faithful unless
you opt in; the GUI exposes each one as a checkbox / drop-down)
------------------------------------------------------------------------------
  * scrape_url: drops boilerplate and duplicate chunks, merges headings into
    the paragraph that follows them (fewer, more distinct chunks).
  * BipolarEncoder.fit / encode_corpus: IDF weighting + corpus-mean centring
    (EXTENSION of the text->pattern stand-in) so that shared vocabulary does
    not make every chunk correlated with every other chunk.
  * DynamicMemoryNetwork(rule="projection"): pseudo-inverse (projection)
    learning for W_A and W_T (EXTENSION; the paper's rule is "hebbian").
  * DynamicMemoryNetwork(alpha<1): leaky / damped synchronous update
    (EXTENSION; alpha=1 is the paper's Eq. 11).
  * Vectorised overlaps, capacity warnings, stability_report(),
    recall_accuracy(), replay_score(), recommend_n(), run_benchmark().
  * GUI: learning-rule drop-down, auto-N, decorrelation, cleaning, damping alpha and
    gate threshold theta_g controls; replay order score; recall verdict distinguishes
    "wrong chunk" from "blended state".
  * Diagnostics tab redrawn each time without stacking colour bars.
  * .ptmem files load with torch.load(weights_only=True) (no pickle code
    execution) and now store the encoder state and learning rule.
  * JSON is shown/saved with real Unicode instead of \\u2019 escapes.

Text -> pattern encoding is a distributed random-projection scheme (a stand-in
for the paper's abstract +-1 patterns xi): every word gets a fixed pseudo-random
+-1 vector, and a chunk's pattern is the (thresholded) sum of its word vectors.
This keeps the whole pipeline dependency-light and fully offline once a page
has been scraped, while preserving the property the paper's patterns need:
similar text lands near similar patterns.

Dependencies
------------
  pip install torch numpy requests beautifulsoup4 matplotlib qtpy PyQt5

Usage
-----
  python memory_as_dynamics.py                 # launch the desktop UI
  python memory_as_dynamics.py --headless URL  # console demo, no UI, needs network
  python memory_as_dynamics.py --selftest      # offline pipeline check, no network/UI
  python memory_as_dynamics.py --benchmark     # results table on a synthetic correlated corpus
  python memory_as_dynamics.py --benchmark URL # same table on a real page (needs network)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import requests
from bs4 import BeautifulSoup


# ============================================================================
# 1. Scraping -> JSON
# ============================================================================

SAMPLE_HTML_FOR_SELFTEST = """
<html><head><title>Sample Page: Neural Memory</title></head>
<body>
  <h1>Attractor Networks Store Memories as Stable States</h1>
  <p>A Hopfield network is a recurrent neural network that stores patterns
  as fixed points of its dynamics, allowing noisy or partial cues to be
  completed back to the original memory.</p>
  <h2>Sequences Are Trajectories</h2>
  <p>Ordered experiences can be encoded as trajectories that visit a series
  of attractor states one after another, driven by asymmetric connections
  learned through temporally asymmetric plasticity.</p>
  <p>A gating signal can switch the network between holding a single memory
  steady and advancing along a learned sequence of memories.</p>
  <h2>Why It Matters For Hardware</h2>
  <p>Because memory and computation share the same weight matrix, this kind
  of model maps naturally onto in-memory and neuromorphic hardware, where
  moving data between separate memory and compute units is costly.</p>
  <p>If you'd like, you can listen to this article here.</p>
  <li>Stable attractors give content-addressable recall from partial cues.</li>
  <li>Temporal association rules give ordered, replayable sequences.</li>
</body></html>
"""

# Page furniture that is not content (audio-player prompts, newsletter boxes, ...).
BOILERPLATE_RE = re.compile(
    r"(listen to this article|subscribe|newsletter|cookie|share (this|on)|follow us|sign up|"
    r"read more|related (articles|posts|stories)|all rights reserved|privacy policy|"
    r"terms of (use|service)|advertisement|skip to (main )?content)",
    re.IGNORECASE,
)
_HEADING_TAGS = ("h1", "h2", "h3", "h4")


def scrape_url(
    url: str,
    max_chunks: int = 48,
    min_chars: int = 20,
    max_chars: int = 400,
    timeout: int = 15,
    html_override: Optional[str] = None,
    clean: bool = True,
    merge_headings: bool = True,
    min_words: int = 5,
) -> Dict:
    """Fetch a URL (or use html_override for offline testing) and pull out an
    ordered list of readable text chunks (headings, paragraphs, list items).

    clean=True         drops boilerplate (BOILERPLATE_RE), near-empty items
                       (< min_words words) and exact duplicates.
    merge_headings=True prepends each heading to the paragraph that follows it
                       instead of storing the heading as a separate, very
                       short (and therefore very generic) chunk.
    clean=False, merge_headings=False reproduces the v1 behaviour.
    """
    if html_override is not None:
        html = html_override
    else:
        headers = {"User-Agent": "Mozilla/5.0 (MemoryAsDynamics/1.0; research tool)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # requests falls back to ISO-8859-1 for text/* without a charset, which
        # garbles curly quotes and dashes; prefer the detected encoding then.
        if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding or "utf-8"
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    chunks: List[Dict] = []
    dropped = 0
    seen = set()
    pending_heading: Optional[str] = None
    order = 0
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if len(text) < min_chars:
            continue
        is_heading = el.name in _HEADING_TAGS

        if clean:
            if BOILERPLATE_RE.search(text):
                dropped += 1
                continue
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            if not is_heading and len(text.split()) < min_words:
                dropped += 1
                continue

        if merge_headings and is_heading:
            pending_heading = text          # attach to the next body chunk
            continue

        section = None
        if merge_headings and pending_heading:
            section = pending_heading
            text = f"{pending_heading}. {text}"
            pending_heading = None
        text = text[:max_chars]

        chunk = {"id": order, "order": order, "tag": el.name, "text": text}
        if section:
            chunk["section"] = section
        chunks.append(chunk)
        order += 1
        if len(chunks) >= max_chunks:
            break

    if not chunks:
        raise ValueError("No readable text chunks (headings/paragraphs/list items) were found on this page.")

    return {"title": title, "chunks": chunks, "dropped": dropped}


def build_json_record(url: str, scraped: Dict) -> Dict:
    """Package scraped content into the JSON record the memory is built from."""
    record = {
        "url": url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "title": scraped["title"],
        "num_chunks": len(scraped["chunks"]),
        "chunks": scraped["chunks"],
    }
    if scraped.get("dropped"):
        record["dropped_chunks"] = scraped["dropped"]
    return record


def dumps_record(record: Dict) -> str:
    """JSON with real Unicode (curly quotes, dashes) instead of \\u2019-style escapes."""
    return json.dumps(record, indent=2, ensure_ascii=False)


# ============================================================================
# 2. Text -> distributed bipolar pattern encoding
# ============================================================================

class BipolarEncoder:
    """Maps text to distributed patterns in {-1, +1}^dim.

    Each token is assigned a fixed pseudo-random +-1 vector (seeded from a
    hash of the token, so the mapping is stable across runs). A chunk's
    pattern is the element-wise sign of the sum of its token vectors -- a
    simple, dependency-free stand-in for the paper's abstract patterns xi,
    chosen so that chunks sharing vocabulary land close together in state
    space (nonzero overlap), while unrelated chunks are close to orthogonal.

    v2 (EXTENSION, opt-in via fit()/encode_corpus()):
      * IDF weighting -- words that occur in most chunks of the page (the
        article's core vocabulary, function words) contribute less.
      * Mean centring -- the average weighted sum over the corpus is
        subtracted before taking the sign, which removes the component that
        every chunk of the page shares.
    Without fit(), encode() behaves exactly as in v1.
    """

    def __init__(self, dim: int = 256, seed: int = 1234):
        self.dim = dim
        self.seed = seed
        self._cache: Dict[str, np.ndarray] = {}
        self.idf: Dict[str, float] = {}
        self.default_idf: float = 1.0
        self.center: Optional[np.ndarray] = None

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9']+", text.lower())

    def _token_vector(self, token: str) -> np.ndarray:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        digest = hashlib.md5(f"{self.seed}:{token}".encode("utf-8")).hexdigest()
        rng = np.random.RandomState(int(digest, 16) % (2**32 - 1))
        vec = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.dim)
        self._cache[token] = vec
        return vec

    def _accumulate(self, tokens: Sequence[str]) -> np.ndarray:
        acc = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens:
            w = self.idf.get(tok, self.default_idf) if self.idf else 1.0
            acc += w * self._token_vector(tok)
        return acc

    # ---- corpus statistics (EXTENSION) --------------------------------------
    def fit(self, texts: Sequence[str], use_idf: bool = True, center: bool = True) -> "BipolarEncoder":
        token_lists = [self._tokens(t) for t in texts]
        n_docs = max(1, len(token_lists))
        if use_idf:
            df = Counter(tok for toks in token_lists for tok in set(toks))
            self.idf = {tok: math.log((1.0 + n_docs) / (1.0 + d)) + 1.0 for tok, d in df.items()}
            self.default_idf = math.log(1.0 + n_docs) + 1.0     # unseen token == maximally rare
        else:
            self.idf, self.default_idf = {}, 1.0
        if center and token_lists:
            self.center = np.stack([self._accumulate(toks) for toks in token_lists]).mean(axis=0)
        else:
            self.center = None
        return self

    def encode_corpus(self, texts: Sequence[str], use_idf: bool = True, center: bool = True) -> List[np.ndarray]:
        """fit() on the page's own chunks, then encode them."""
        self.fit(texts, use_idf=use_idf, center=center)
        return [self.encode(t) for t in texts]

    # ---- encoding -------------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        tokens = self._tokens(text)
        if not tokens:
            digest = hashlib.md5(f"{self.seed}:{text}".encode("utf-8")).hexdigest()
            rng = np.random.RandomState(int(digest, 16) % (2**32 - 1))
            return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.dim)
        acc = self._accumulate(tokens)
        if self.center is not None:
            acc = acc - self.center
        pattern = np.sign(acc)
        pattern[pattern == 0.0] = 1.0
        return pattern.astype(np.float32)

    @staticmethod
    def corrupt(pattern: np.ndarray, noise_frac: float, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Flip a fraction of bits -- used to build the noisy cues of Task A."""
        rng = rng if rng is not None else np.random.RandomState()
        out = pattern.copy()
        n_flip = int(round(max(0.0, min(1.0, noise_frac)) * len(pattern)))
        if n_flip > 0:
            idx = rng.choice(len(pattern), size=n_flip, replace=False)
            out[idx] = out[idx] * -1.0
        return out

    # ---- persistence ----------------------------------------------------------
    def to_state(self) -> Dict:
        """Only plain Python types, so it round-trips through torch.load(weights_only=True)."""
        return {
            "dim": self.dim, "seed": self.seed,
            "idf": dict(self.idf), "default_idf": float(self.default_idf),
            "center": None if self.center is None else [float(v) for v in self.center],
        }

    @classmethod
    def from_state(cls, d: Dict) -> "BipolarEncoder":
        enc = cls(dim=d["dim"], seed=d.get("seed", 1234))
        enc.idf = dict(d.get("idf", {}))
        enc.default_idf = float(d.get("default_idf", 1.0))
        c = d.get("center")
        enc.center = None if c is None else np.asarray(c, dtype=np.float32)
        return enc


# ============================================================================
# 3. The recurrent memory network (paper's core model)
# ============================================================================

# Empirical N-per-chunk ratios from synthetic article-like corpora (P = 40) with the decorrelated
# encoder. Rules of thumb, not guarantees -- re-run --benchmark on your own page.
#   hebbian    : recall@25% noise reaches ~90-100% around N/P = 25; sequence replay never worked.
#   projection : recall is 100% from N/P ~ 6; a full in-order replay with a narrow gate (theta_g=0.9)
#                succeeded on 8/8 corpora from N/P = 9.6, but only 5-6/8 at N/P = 6.4-8.
# With the raw v1 encoder no N was enough on those corpora (see the benchmark table).
HEBBIAN_N_PER_PATTERN = 25
PROJECTION_N_PER_PATTERN = 10


def recommend_n(num_patterns: int, rule: str = "hebbian", max_n: int = 2048, multiple: int = 64) -> int:
    """Rule-of-thumb network size for P stored chunks (multiple of `multiple`, capped at max_n).

    hebbian    : the paper's bounds -- 0.14N >= P (Eq. 14) and N/(3 ln N) >= P-1 (Eq. 15, c=3) --
                 and at least HEBBIAN_N_PER_PATTERN * P, because correlated text needs headroom
                 well beyond the random-pattern formulas.
    projection : N >= PROJECTION_N_PER_PATTERN * P.
    """
    p = max(1, int(num_patterns))
    n = multiple
    while n < max_n:
        if rule == "projection":
            ok = n >= PROJECTION_N_PER_PATTERN * p
        else:
            ok = ((0.14 * n >= p) and (n / (3.0 * math.log(n)) >= p - 1)
                  and n >= HEBBIAN_N_PER_PATTERN * p)
        if ok:
            break
        n += multiple
    return min(n, max_n)


class DynamicMemoryNetwork:
    """Recurrent attractor + sequence memory.

        W = W_A + g(t) * W_T                                   (Eq. 11)

    W_A : symmetric Hebbian matrix, holds items as stable attractors (Eq. 5/8, Sec 4.5)
    W_T : asymmetric successor-predecessor matrix, advances sequences (Eq. 6/9)
    g(t): periodic theta-like gate scaling W_T (Eq. 12)

    rule="hebbian"    paper-faithful outer-product learning (default).
    rule="projection" EXTENSION: pseudo-inverse learning. W_A = Xi^T (Xi Xi^T/N + ridge I)^-1 Xi / N
                      is the projector onto span{xi^mu}; W_T maps each xi^mu exactly onto its
                      successor. It removes the crosstalk between correlated patterns that
                      limits the Hebbian rule (needs P <= N).
    alpha < 1         EXTENSION: leaky update x <- (1-alpha) x + alpha tanh(...) which damps the
                      2-cycles that fully synchronous updates can produce. alpha=1 is Eq. 11.
    """

    def __init__(
        self,
        n_neurons: int,
        beta: float = 2.0,
        gamma: float = 1e-4,
        gmax: float = 1.6,
        period: int = 12,
        theta_g: float = 0.0,
        kappa: float = 8.0,
        device: str = "cpu",
        rule: str = "hebbian",
        ridge: float = 1e-3,
        alpha: float = 1.0,
    ):
        if rule not in ("hebbian", "projection"):
            raise ValueError("rule must be 'hebbian' or 'projection'")
        self.N = n_neurons
        self.beta = beta
        self.gamma = gamma
        self.gmax = gmax
        self.period = period
        self.theta_g = theta_g
        self.kappa = kappa
        self.device = torch.device(device)
        self.rule = rule
        self.ridge = ridge
        self.alpha = alpha

        self.W_A = torch.zeros(self.N, self.N, device=self.device)
        self.W_T = torch.zeros(self.N, self.N, device=self.device)
        self.b = torch.zeros(self.N, device=self.device)

        self.patterns: List[torch.Tensor] = []
        self.labels: List[str] = []
        self._seq_order: List[int] = []
        self._Xi: Optional[torch.Tensor] = None       # (P, N) stacked patterns, for vectorised overlaps
        self._dual: Optional[torch.Tensor] = None     # (P, N) dual basis (projection rule only)

    # ---- storage / learning -------------------------------------------------
    def store_patterns(self, patterns: List[np.ndarray], labels: Optional[List[str]] = None) -> None:
        """Auto-associative storage.

        hebbian    : closed-form batch Hebbian rule
                     W_A = (1/N) * sum_mu xi^mu (xi^mu)^T, with homeostatic decay (Eq. 5, 7, 8; Sec 4.5).
        projection : W_A = Xi^T dual, dual = (G + ridge I)^-1 Xi / N, G = Xi Xi^T / N  (EXTENSION)."""
        if not patterns:
            raise ValueError("No patterns to store.")
        self.patterns = [torch.tensor(p, dtype=torch.float32, device=self.device) for p in patterns]
        self.labels = labels if labels is not None else [f"pattern_{i}" for i in range(len(patterns))]
        Xi = torch.stack(self.patterns)                 # (P, N)
        self._Xi = Xi
        n_pat = Xi.shape[0]
        if self.rule == "projection":
            if n_pat > self.N:
                raise ValueError(
                    f"Projection rule needs P <= N (got P={n_pat}, N={self.N}). Increase N or store fewer chunks."
                )
            gram = (Xi @ Xi.t()) / float(self.N)
            eye = torch.eye(n_pat, device=self.device)
            self._dual = torch.linalg.solve(gram + self.ridge * eye, Xi) / float(self.N)   # (P, N)
            W_A = Xi.t() @ self._dual                   # (N, N) projector onto span(Xi)
        else:
            self._dual = None
            W_A = (Xi.t() @ Xi) / float(self.N)          # Eq. (5)/(8) closed form
            W_A.fill_diagonal_(0.0)                      # standard no-self-connection variant
        self.W_A = W_A * (1.0 - self.gamma)              # Eq. (7)

    def store_sequence(self, order: Optional[List[int]] = None, cyclic: bool = False) -> None:
        """Hetero-associative sequence storage.

        hebbian    : Eq. (9)  W_T = (1/N) * sum_{i=1}^{k-1} x_{i+1} x_i^T (successor-predecessor).
        projection : W_T = sum_i x_{i+1} dual_i^T, so W_T xi_i = xi_{i+1} exactly (EXTENSION)."""
        if len(self.patterns) < 2:
            self.W_T = torch.zeros(self.N, self.N, device=self.device)
            self._seq_order = list(range(len(self.patterns)))
            return
        idx = order if order is not None else list(range(len(self.patterns)))
        seq = torch.stack([self.patterns[i] for i in idx])          # (k, N)
        if self.rule == "projection":
            assert self._dual is not None, "store_patterns() must run before store_sequence()"
            W_T = seq[1:].t() @ self._dual[idx[:-1]]
            if cyclic:
                W_T = W_T + torch.outer(seq[0], self._dual[idx[-1]])
        else:
            W_T = seq[1:].t() @ seq[:-1]
            if cyclic:
                W_T = W_T + torch.outer(seq[0], seq[-1])
            W_T = W_T / float(self.N)
        self.W_T = W_T * (1.0 - self.gamma)
        self._seq_order = list(idx)

    # ---- dynamics -------------------------------------------------------------
    def gate(self, t: float) -> float:
        """Periodic theta-like gate, Eq. (12): g(t) = gmax * sigmoid(kappa*(cos(2*pi*t/T) - theta_g))."""
        z = self.kappa * (math.cos(2.0 * math.pi * t / self.period) - self.theta_g)
        return float(self.gmax / (1.0 + math.exp(-z)))

    def step(self, x: torch.Tensor, gate_val: float = 0.0, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One update of Eq. (11): x(t+1) = tanh(beta * (W_A x + g(t) W_T x + U u + b))."""
        a = self.W_A @ x + gate_val * (self.W_T @ x) + self.b
        if u is not None:
            a = a + u
        x_new = torch.tanh(self.beta * a)
        if self.alpha < 1.0:
            x_new = (1.0 - self.alpha) * x + self.alpha * x_new
        return x_new

    def energy(self, x: torch.Tensor) -> float:
        """Hopfield-style energy of the symmetric (attractor-only) part, Sec 3.1.
        Not defined once the asymmetric W_T term is active (no Lyapunov function then, Sec 3.3/8)."""
        return float(-0.5 * (x @ (self.W_A @ x)) - (x @ self.b))

    def overlap(self, x: torch.Tensor, pattern: torch.Tensor) -> float:
        """m = xi . x / (||xi|| ||x||), Table 2."""
        denom = (x.norm() * pattern.norm()).clamp_min(1e-8)
        return float((pattern @ x) / denom)

    def overlaps(self, x: torch.Tensor) -> torch.Tensor:
        """Vector of overlaps of x with every stored pattern (one matmul instead of P python calls)."""
        if self._Xi is None:
            return torch.zeros(0, device=self.device)
        denom = (x.norm() * self._Xi.norm(dim=1)).clamp_min(1e-8)
        return (self._Xi @ x) / denom

    def best_match(self, x: torch.Tensor) -> Tuple[int, float]:
        if not self.patterns:
            return -1, 0.0
        ov = self.overlaps(x)
        i = int(torch.argmax(ov))
        return i, float(ov[i])

    def overlap_matrix(self) -> np.ndarray:
        """(P, P) matrix xi^mu . xi^nu / N. Off-diagonal entries are the crosstalk that limits capacity."""
        if self._Xi is None:
            return np.zeros((0, 0), dtype=np.float32)
        return ((self._Xi @ self._Xi.t()) / float(self.N)).cpu().numpy()

    # ---- recall (Task A) --------------------------------------------------
    def recall(self, cue: np.ndarray, steps: int = 60, gate_val: float = 0.0, tol: float = 1e-4) -> Dict:
        """Free-run retrieval from a (possibly noisy) cue, Eq. (1)/(11), Task A."""
        x = torch.tensor(cue, dtype=torch.float32, device=self.device)
        overlap_trace: List[float] = []
        energy_trace: List[float] = []
        converged_at: Optional[int] = None
        for t in range(steps):
            _, best_ov = self.best_match(x)
            overlap_trace.append(best_ov)
            energy_trace.append(self.energy(x))
            x_next = self.step(x, gate_val=gate_val)
            if converged_at is None and torch.norm(x_next - x) < tol:
                converged_at = t
            x = x_next
        final_idx, final_ov = self.best_match(x)
        return {
            "final_state": x.cpu().numpy(),
            "final_best_idx": final_idx,
            "final_overlap": final_ov,
            "overlap_trace": overlap_trace,
            "energy_trace": energy_trace,
            "converged_at": converged_at,
        }

    # ---- sequence replay (Task B) ------------------------------------------
    def replay_sequence(self, steps: int = 60, x0: Optional[torch.Tensor] = None) -> Dict:
        """Free-run the gated dynamics starting from (by default) the first stored
        item, Eq. (11)-(12), Task B."""
        if x0 is None:
            if not self.patterns:
                raise ValueError("No patterns stored.")
            start_idx = self._seq_order[0] if self._seq_order else 0
            x0 = self.patterns[start_idx].clone()
        x = x0.clone()
        gate_trace: List[float] = []
        index_trace: List[int] = []
        overlap_trace: List[float] = []
        for t in range(steps):
            g = self.gate(t)
            gate_trace.append(g)
            idx, ov = self.best_match(x)
            index_trace.append(idx)
            overlap_trace.append(ov)
            x = self.step(x, gate_val=g)
        return {"gate_trace": gate_trace, "index_trace": index_trace, "overlap_trace": overlap_trace}

    def score_replay(self, index_trace: Sequence[int]) -> Dict:
        """How faithfully did the replay follow the stored order?

        visited          best-matching chunk each time it changed
        transitions      number of changes
        correct          changes that went to the stored successor
        in_order_prefix  correct transitions before the first error
        advances_per_cycle  chunk changes per gate period (~1 for a theta-like one-item-per-cycle replay)
        completed / steps_to_end  True if the whole stored sequence was traversed in order, and the step
                         at which the last item was reached; anything after that first arrival (the
                         sequence has no successor there) is ignored
        """
        order = self._seq_order or list(range(len(self.patterns)))
        if not order:
            return {"completed": False, "steps_to_end": None, "visited": [], "transitions": 0,
                    "correct": 0, "in_order_prefix": 0, "advances_per_cycle": 0.0}
        pos = {p: i for i, p in enumerate(order)}
        trace = list(index_trace)
        if order[-1] in trace:
            trace = trace[: trace.index(order[-1]) + 1]
        visited = [idx for k, idx in enumerate(trace) if k == 0 or idx != trace[k - 1]]
        pairs = list(zip(visited, visited[1:]))
        ok = [(a in pos and b in pos and pos[b] == pos[a] + 1) for a, b in pairs]
        prefix = 0
        for flag in ok:
            if not flag:
                break
            prefix += 1
        cycles = max(len(trace) / float(self.period), 1e-9)
        completed = prefix == len(order) - 1
        return {
            "completed": completed,
            "steps_to_end": len(trace) if completed else None,
            "visited": visited,
            "transitions": len(pairs),
            "correct": int(sum(ok)),
            "in_order_prefix": prefix,
            "advances_per_cycle": len(pairs) / cycles,
        }

    def replay_score(self, steps: Optional[int] = None) -> Dict:
        """Run a replay long enough to traverse the stored sequence and score it."""
        n_items = max(len(self._seq_order), 2)
        steps = steps if steps is not None else self.period * n_items
        res = self.replay_sequence(steps=steps)
        out = self.score_replay(res["index_trace"])
        out["sequence_length"] = n_items
        return out

    # ---- diagnostics (Eqs. 3-4, 14-16) -------------------------------------
    def spectral_radius_at(self, x: torch.Tensor) -> float:
        """rho(J(x*)) for the symmetric part, Eq. (3)-(4). Should be < 1 at a stable memory.

        J = diag(phi') W_A with phi' >= 0 and W_A symmetric, so J has the same spectrum as the
        symmetric matrix diag(sqrt(phi')) W_A diag(sqrt(phi')). That allows eigvalsh (small N)
        or plain power iteration (large N) instead of a full complex eigendecomposition."""
        a = self.W_A @ x + self.b
        phi_prime = self.beta * (1.0 - torch.tanh(self.beta * a) ** 2)
        s = torch.sqrt(phi_prime.clamp_min(0.0))
        M = s[:, None] * self.W_A * s[None, :]
        if self.N <= 768:
            return float(torch.linalg.eigvalsh(M).abs().max())
        gen = torch.Generator().manual_seed(0)
        v = torch.randn(self.N, generator=gen).to(self.device)
        v = v / v.norm()
        rho = 0.0
        for _ in range(300):
            w = M @ v
            rho = float(w.norm())
            if rho < 1e-12:
                return 0.0
            v = w / w.norm()
        return rho

    def capacity_estimates(self) -> Dict[str, float]:
        """Pmax ~ 0.14N (Eq. 14) and Smax ~ N/(c ln N), c in [2,4] (Eq. 15)."""
        n = max(self.N, 3)
        return {
            "Pmax_hebbian": 0.14 * n,
            "Smax_low_c4": n / (4.0 * math.log(n)),
            "Smax_high_c2": n / (2.0 * math.log(n)),
        }

    def capacity_warning(self) -> Optional[str]:
        """Human-readable warning if the stored load exceeds the model's own capacity estimates."""
        n_pat = len(self.patterns)
        if n_pat == 0:
            return None
        caps = self.capacity_estimates()
        msgs: List[str] = []
        if self.rule == "hebbian":
            if n_pat > caps["Pmax_hebbian"]:
                msgs.append(
                    f"P = {n_pat} exceeds the Hebbian pattern capacity 0.14N = {caps['Pmax_hebbian']:.1f} (Eq. 14): "
                    f"expect blended or wrong-neighbour recall."
                )
            if len(self._seq_order) > caps["Smax_high_c2"]:
                msgs.append(
                    f"Stored sequence length {len(self._seq_order)} exceeds Smax ~ N/(c ln N) = "
                    f"[{caps['Smax_low_c4']:.1f}, {caps['Smax_high_c2']:.1f}] (Eq. 15): replay will stall or skip."
                )
        elif n_pat > 0.5 * self.N:
            msgs.append(f"P = {n_pat} > N/2: projection-rule basins of attraction get very small; increase N.")
        return " ".join(msgs) if msgs else None

    def energy_proxy(self, traj: List[torch.Tensor], e_spike: float = 1.0) -> float:
        """Event-based energy proxy, Eq. (16), with firing probability s_i = (1+x_i)/2."""
        total = 0.0
        for x in traj:
            s = (1.0 + x) / 2.0
            total += float(s.sum()) * e_spike
        return total

    def recall_accuracy(self, noise: float = 0.25, trials: int = 1, steps: int = 60, seed: int = 0) -> Dict:
        """Basin test: cue every stored chunk with `noise` bits flipped and check what comes back.

        accuracy      fraction of cues whose best-matching chunk is the target
        clean         fraction that ALSO ended in a clean state (overlap >= 0.9)
        mean_overlap  mean final overlap with the best-matching chunk"""
        rng = np.random.RandomState(seed)
        hits = clean = 0
        ov_sum = 0.0
        total = 0
        for mu, pat in enumerate(self.patterns):
            base = pat.cpu().numpy()
            for _ in range(trials):
                res = self.recall(BipolarEncoder.corrupt(base, noise, rng=rng), steps=steps)
                hit = res["final_best_idx"] == mu
                hits += int(hit)
                clean += int(hit and res["final_overlap"] >= 0.9)
                ov_sum += res["final_overlap"]
                total += 1
        total = max(total, 1)
        return {"accuracy": hits / total, "clean": clean / total, "mean_overlap": ov_sum / total, "trials": total}

    def stability_report(self, noise_levels: Sequence[float] = (0.0, 0.10, 0.25), steps: int = 60) -> Dict:
        """Everything the Diagnostics tab shows, in one call."""
        n_pat = len(self.patterns)
        caps = self.capacity_estimates()
        corr = self.overlap_matrix()
        off = corr[~np.eye(n_pat, dtype=bool)] if n_pat > 1 else np.zeros(1)
        rhos = np.array([self.spectral_radius_at(p) for p in self.patterns]) if n_pat else np.zeros(0)
        report = {
            "N": self.N, "P": n_pat, "load": n_pat / float(self.N), "rule": self.rule,
            "Pmax_hebbian": caps["Pmax_hebbian"],
            "mean_abs_overlap": float(np.abs(off).mean()), "max_abs_overlap": float(np.abs(off).max()),
            "rho": rhos, "rho0": float(rhos[0]) if n_pat else float("nan"),
            "frac_stable": float((rhos < 1.0).mean()) if n_pat else float("nan"),
            "recall": {f"{noise:.2f}": self.recall_accuracy(noise, steps=steps) for noise in noise_levels},
            "warning": self.capacity_warning(),
        }
        return report

    # ---- persistence --------------------------------------------------------
    def to_state(self) -> Dict:
        return {
            "N": self.N, "beta": self.beta, "gamma": self.gamma, "gmax": self.gmax,
            "period": self.period, "theta_g": self.theta_g, "kappa": self.kappa,
            "rule": self.rule, "ridge": self.ridge, "alpha": self.alpha,
            "W_A": self.W_A.cpu(), "W_T": self.W_T.cpu(), "b": self.b.cpu(),
            "patterns": [p.cpu() for p in self.patterns], "labels": self.labels,
            "seq_order": self._seq_order,
        }

    @classmethod
    def from_state(cls, d: Dict) -> "DynamicMemoryNetwork":
        net = cls(d["N"], beta=d["beta"], gamma=d["gamma"], gmax=d["gmax"],
                   period=d["period"], theta_g=d["theta_g"], kappa=d["kappa"],
                   rule=d.get("rule", "hebbian"), ridge=d.get("ridge", 1e-3), alpha=d.get("alpha", 1.0))
        net.W_A = d["W_A"]
        net.W_T = d["W_T"]
        net.b = d["b"]
        net.patterns = d["patterns"]
        net.labels = d["labels"]
        net._seq_order = d.get("seq_order", list(range(len(d["labels"]))))
        net._Xi = torch.stack(net.patterns) if net.patterns else None
        return net


# ============================================================================
# 4. Offline self-test, benchmark, headless demo (no UI)
# ============================================================================

def build_network(
    texts: Sequence[str],
    n_neurons: int,
    rule: str = "hebbian",
    decorrelate: bool = False,
    use_sequence: bool = True,
    alpha: float = 1.0,
    theta_g: float = 0.0,
) -> Tuple[DynamicMemoryNetwork, BipolarEncoder]:
    """encode -> store attractors -> store reading-order sequence."""
    encoder = BipolarEncoder(dim=n_neurons)
    patterns = encoder.encode_corpus(texts) if decorrelate else [encoder.encode(t) for t in texts]
    labels = [f"{t[:60]}" for t in texts]
    net = DynamicMemoryNetwork(n_neurons=n_neurons, rule=rule, alpha=alpha, theta_g=theta_g)
    net.store_patterns(patterns, labels)
    if use_sequence:
        net.store_sequence()
    return net, encoder


def run_pipeline(
    url: str,
    n_neurons: int = 256,
    html_override: Optional[str] = None,
    rule: str = "hebbian",
    decorrelate: bool = False,
    auto_n: bool = False,
    clean: bool = True,
) -> Dict:
    """Run scrape -> JSON -> encode -> store -> recall -> replay end to end."""
    scraped = scrape_url(url, html_override=html_override, clean=clean)
    record = build_json_record(url, scraped)
    texts = [c["text"] for c in record["chunks"]]
    if auto_n:
        n_neurons = recommend_n(len(texts), rule)

    net, encoder = build_network(texts, n_neurons, rule=rule, decorrelate=decorrelate)
    net.labels = [f"[{c['tag']}] {c['text'][:60]}" for c in record["chunks"]]

    rng = np.random.RandomState(0)
    cue = BipolarEncoder.corrupt(net.patterns[0].cpu().numpy(), 0.25, rng=rng)
    recall_result = net.recall(cue, steps=40)
    replay_result = net.replay_sequence(steps=min(30, 3 * len(texts)))

    return {
        "record": record,
        "network": net,
        "encoder": encoder,
        "recall_result": recall_result,
        "replay_result": replay_result,
    }


def synthetic_corpus(num_chunks: int = 40, seed: int = 7) -> List[str]:
    """A deterministic stand-in for a single-topic web article.

    Real articles share a lot of vocabulary from start to finish (stop-words plus the article's
    own key terms), which is exactly what makes their chunk patterns correlated. Each synthetic
    chunk is 30 tokens: ~30% common stop-words, ~30% article-wide key terms, ~25% words from the
    sub-topic of its section (6 chunks per section) and ~15% rare words."""
    rng = np.random.RandomState(seed)
    stop = [f"stop{i}" for i in range(25)]
    key = [f"key{i}" for i in range(20)]
    sub = [[f"sub{s}w{i}" for i in range(40)] for s in range(max(1, num_chunks // 6 + 1))]
    rare = [f"rare{i}" for i in range(400)]
    texts = []
    for c in range(num_chunks):
        toks = (
            list(rng.choice(stop, 9)) + list(rng.choice(key, 9))
            + list(rng.choice(sub[c // 6], 8)) + list(rng.choice(rare, 4))
        )
        rng.shuffle(toks)
        texts.append(" ".join(toks))
    return texts


def run_benchmark(
    texts: Sequence[str],
    sizes: Sequence[int] = (128, 256, 512, 1024),
    noise_levels: Sequence[float] = (0.0, 0.10, 0.25),
    label: str = "",
) -> List[Dict]:
    """Recall / stability / replay table: paper setup vs. each opt-in extension, for several N."""
    configs = [
        ("paper (Hebbian, raw encoder)", dict(rule="hebbian", decorrelate=False)),
        ("Hebbian + decorrelated encoder", dict(rule="hebbian", decorrelate=True)),
        ("  ... + damping alpha=0.25", dict(rule="hebbian", decorrelate=True, alpha=0.25)),
        ("Projection + decorrelated encoder", dict(rule="projection", decorrelate=True)),
        ("  ... + narrow gate theta_g=0.9", dict(rule="projection", decorrelate=True, theta_g=0.9)),
    ]
    rows: List[Dict] = []
    head = (f"{'configuration':<36}{'N':>5}  " + " ".join(f"rec@{int(nz*100):>2}%" for nz in noise_levels)
            + "  stable  mean|xi.xi'|  replay in-order  steps/chunk")
    print(f"\nBenchmark {label}: P = {len(texts)} chunks.")
    print("rec@x% = share of chunks recovered from a cue with x% of bits flipped (0% = are stored chunks fixed points at all?).")
    print(head)
    print("-" * len(head))
    for name, kw in configs:
        for n in sizes:
            if kw["rule"] == "projection" and len(texts) > n:
                continue
            net, _ = build_network(texts, n, **kw)
            rep = net.stability_report(noise_levels=noise_levels)
            sc = net.replay_score()
            row = {
                "config": name.strip(), "N": n, "stable": rep["frac_stable"], "mean_abs_overlap": rep["mean_abs_overlap"],
                "recall": {k: v["accuracy"] for k, v in rep["recall"].items()},
                "in_order_prefix": sc["in_order_prefix"], "seq_len": sc["sequence_length"],
                "completed": sc["completed"], "steps_to_end": sc["steps_to_end"],
            }
            rows.append(row)
            rec = " ".join(f"{row['recall'][k]*100:5.0f}%" for k in row["recall"])
            spc = (f"{sc['steps_to_end'] / max(sc['sequence_length'] - 1, 1):>8.1f}" if sc["completed"] else "       -")
            print(f"{name:<36}{n:>5}  {rec}  {row['stable']*100:5.0f}%  {row['mean_abs_overlap']:11.3f}  "
                  f"{sc['in_order_prefix']:>7}/{sc['sequence_length']-1:<4}  {spc}")
    print("stable = share of chunks with rho(J) < 1 (Eq. 3-4). replay in-order = correct successive chunks before the first error;"
          " steps/chunk = time to traverse the sequence (12 = one chunk per gate cycle), '-' = never reached the end.")
    return rows


def selftest() -> None:
    print("Running offline self-test (embedded sample HTML, no network) ...")
    out = run_pipeline("about:sample", n_neurons=192, html_override=SAMPLE_HTML_FOR_SELFTEST)
    record, net = out["record"], out["network"]
    print(f"Title: {record['title']}")
    print(f"Chunks stored: {record['num_chunks']}   (boilerplate/duplicates dropped: {record.get('dropped_chunks', 0)})")
    print(dumps_record({"chunks": record["chunks"][:2]}))
    rr = out["recall_result"]
    print(f"\nRecall from a 25%-noise cue of chunk 0:")
    print(f"  converged at step {rr['converged_at']}, final best match = "
          f"chunk {rr['final_best_idx']} (overlap {rr['final_overlap']:.3f})")
    rep = out["replay_result"]
    print(f"\nSequence replay index trace (first 10 steps): {rep['index_trace'][:10]}")
    caps = net.capacity_estimates()
    print(f"\nCapacity estimates for N={net.N}: {caps}")
    warn = net.capacity_warning()
    print(f"Capacity warning: {warn or 'none'}")

    # extension smoke checks -------------------------------------------------
    texts = [c["text"] for c in record["chunks"]]
    net_p, _ = build_network(texts, 128, rule="projection", decorrelate=True)
    acc = net_p.recall_accuracy(0.10)["accuracy"]
    sc = net_p.replay_score()
    print(f"Projection rule on the sample page: recall@10% = {acc:.0%}, "
          f"replay in-order = {sc['in_order_prefix']}/{sc['sequence_length'] - 1}")
    print("\nSelf-test OK.")


def run_headless(url: str, n_neurons: int = 256, rule: str = "hebbian", decorrelate: bool = False, auto_n: bool = False) -> None:
    out = run_pipeline(url, n_neurons=n_neurons, rule=rule, decorrelate=decorrelate, auto_n=auto_n)
    record, net = out["record"], out["network"]
    print(f"Title: {record['title']}")
    print(f"Chunks stored: {record['num_chunks']}   N = {net.N}   rule = {net.rule}")
    print(dumps_record(record)[:2000])
    rr = out["recall_result"]
    print(f"\nRecall demo: converged_at={rr['converged_at']}, "
          f"final_best_idx={rr['final_best_idx']}, overlap={rr['final_overlap']:.3f}")
    print(f"Capacity estimates: {net.capacity_estimates()}")
    warn = net.capacity_warning()
    if warn:
        print(f"WARNING: {warn}")


# ============================================================================
# 5. QtPy desktop UI
# ============================================================================

def _launch_gui() -> int:
    from qtpy.QtCore import Qt, QThread, Signal
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
        QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QSlider,
        QTabWidget, QListWidget, QPlainTextEdit, QFileDialog, QMessageBox,
        QGroupBox, QFormLayout, QCheckBox, QStatusBar, QComboBox,
    )

    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    # ---- background worker for scraping + building (keeps UI responsive) --
    class BuildWorker(QThread):
        progress = Signal(str)
        finished_ok = Signal(dict)
        failed = Signal(str)

        def __init__(self, url: str, n_neurons: int, use_sequence: bool,
                     rule: str = "hebbian", auto_n: bool = False,
                     decorrelate: bool = False, clean: bool = True, alpha: float = 1.0):
            super().__init__()
            self.url = url
            self.n_neurons = n_neurons
            self.use_sequence = use_sequence
            self.rule = rule
            self.auto_n = auto_n
            self.decorrelate = decorrelate
            self.clean = clean
            self.alpha = alpha

        def run(self):
            try:
                self.progress.emit(f"Fetching {self.url} ...")
                scraped = scrape_url(self.url, clean=self.clean, merge_headings=self.clean)
                dropped = scraped.get("dropped", 0)
                extra = f" ({dropped} boilerplate/duplicate items dropped)" if dropped else ""
                self.progress.emit(f"Found {len(scraped['chunks'])} text chunks{extra}. Building JSON record ...")
                record = build_json_record(self.url, scraped)
                texts = [c["text"] for c in record["chunks"]]

                n = self.n_neurons
                if self.auto_n:
                    n = recommend_n(len(texts), self.rule)
                    self.progress.emit(f"Auto-sized N = {n} for P = {len(texts)} chunks ({self.rule} rule).")

                enc_note = " (IDF-weighted, mean-centred)" if self.decorrelate else ""
                self.progress.emit(f"Encoding chunks into {n}-dim bipolar patterns{enc_note} ...")
                encoder = BipolarEncoder(dim=n)
                patterns = encoder.encode_corpus(texts) if self.decorrelate else [encoder.encode(t) for t in texts]
                labels = [f"[{c['tag']}] {c['text'][:60]}" for c in record["chunks"]]

                rule_note = "projection rule, extension" if self.rule == "projection" else "Hebbian rule, Eq. 5/8"
                self.progress.emit(f"Storing attractor memories ({rule_note}) ...")
                net = DynamicMemoryNetwork(n_neurons=n, rule=self.rule, alpha=self.alpha)
                net.store_patterns(patterns, labels)

                if self.use_sequence:
                    self.progress.emit("Storing reading-order sequence (temporal rule, Eq. 9) ...")
                    net.store_sequence()

                self.progress.emit("Memory built successfully.")
                self.finished_ok.emit({"record": record, "network": net, "encoder": encoder})
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    class MplCanvas(FigureCanvas):
        def __init__(self, nrows=1, ncols=1, width=5.5, height=3.6, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
            self.axes = self.fig.subplots(nrows, ncols)
            super().__init__(self.fig)

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Memory as Dynamics — URL Memory Encoder")
            self.resize(1180, 900)

            self.record: Optional[Dict] = None
            self.network: Optional[DynamicMemoryNetwork] = None
            self.encoder: Optional[BipolarEncoder] = None
            self.worker: Optional[BuildWorker] = None

            self._build_ui()

        # -------------------------------------------------------------- UI
        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            root = QVBoxLayout(central)

            # -- top control bar --------------------------------------------------
            top_box = QGroupBox("Build memory from a URL")
            top_outer = QVBoxLayout(top_box)
            top_layout = QHBoxLayout()
            top_outer.addLayout(top_layout)

            top_layout.addWidget(QLabel("URL:"))
            self.url_edit = QLineEdit("https://en.wikipedia.org/wiki/Hopfield_network")
            self.url_edit.setMinimumWidth(360)
            top_layout.addWidget(self.url_edit, 3)

            top_layout.addWidget(QLabel("Neurons N:"))
            self.n_spin = QSpinBox()
            self.n_spin.setRange(32, 2048)
            self.n_spin.setSingleStep(32)
            self.n_spin.setValue(256)
            top_layout.addWidget(self.n_spin)

            self.seq_check = QCheckBox("Store reading-order sequence")
            self.seq_check.setChecked(True)
            top_layout.addWidget(self.seq_check)

            self.build_btn = QPushButton("Scrape && Build Memory")
            self.build_btn.clicked.connect(self.on_build_clicked)
            top_layout.addWidget(self.build_btn)

            opts = QHBoxLayout()
            top_outer.addLayout(opts)
            opts.addWidget(QLabel("Learning rule:"))
            self.rule_combo = QComboBox()
            self.rule_combo.addItem("Hebbian (paper, Eq. 5/8/9)", "hebbian")
            self.rule_combo.addItem("Projection / pseudo-inverse (extension)", "projection")
            opts.addWidget(self.rule_combo)
            self.auto_n_check = QCheckBox("Auto-size N for chunk count")
            self.auto_n_check.setToolTip("Pick N from the capacity estimates (Eq. 14/15) once the page has been scraped.")
            opts.addWidget(self.auto_n_check)
            self.decor_check = QCheckBox("Decorrelate encoding (IDF + centring)")
            self.decor_check.setToolTip("Down-weights words shared by most chunks and removes the page-wide mean pattern (extension).")
            opts.addWidget(self.decor_check)
            self.clean_check = QCheckBox("Clean page (drop boilerplate, merge headings)")
            self.clean_check.setChecked(True)
            opts.addWidget(self.clean_check)
            opts.addWidget(QLabel("Update damping \u03b1:"))
            self.alpha_spin = QDoubleSpinBox()
            self.alpha_spin.setRange(0.05, 1.0)
            self.alpha_spin.setSingleStep(0.05)
            self.alpha_spin.setDecimals(2)
            self.alpha_spin.setValue(1.0)
            self.alpha_spin.setToolTip("1.0 = paper's Eq. 11. Below 1 the synchronous update is damped, "
                                       "x <- (1-\u03b1) x + \u03b1 tanh(...) (extension).")
            opts.addWidget(self.alpha_spin)
            opts.addStretch(1)

            root.addWidget(top_box)

            # -- log console -------------------------------------------------------
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(500)
            self.log.setFixedHeight(90)
            mono = QFont("Monospace")
            mono.setStyleHint(QFont.TypeWriter)
            self.log.setFont(mono)
            root.addWidget(self.log)

            # -- tabs ----------------------------------------------------------------
            self.tabs = QTabWidget()
            root.addWidget(self.tabs, 1)

            self.tabs.addTab(self._build_chunks_tab(), "Chunks / JSON")
            self.tabs.addTab(self._build_recall_tab(), "Attractor Recall")
            self.tabs.addTab(self._build_sequence_tab(), "Sequence Replay")
            self.tabs.addTab(self._build_diagnostics_tab(), "Network Diagnostics")

            self.status = QStatusBar()
            self.setStatusBar(self.status)
            self.status.showMessage("Enter a URL and click 'Scrape & Build Memory' to begin.")

        def _build_chunks_tab(self) -> QWidget:
            w = QWidget()
            layout = QHBoxLayout(w)
            splitter = QSplitter(Qt.Horizontal)

            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.addWidget(QLabel("Stored chunks (order = reading order):"))
            self.chunk_list = QListWidget()
            left_layout.addWidget(self.chunk_list)
            splitter.addWidget(left)

            right = QWidget()
            right_layout = QVBoxLayout(right)
            right_layout.addWidget(QLabel("JSON record built from the page:"))
            self.json_view = QPlainTextEdit()
            self.json_view.setReadOnly(True)
            right_layout.addWidget(self.json_view)
            btn_row = QHBoxLayout()
            self.save_json_btn = QPushButton("Save JSON...")
            self.save_json_btn.clicked.connect(self.on_save_json)
            self.save_mem_btn = QPushButton("Save Memory (.ptmem)...")
            self.save_mem_btn.clicked.connect(self.on_save_memory)
            self.load_mem_btn = QPushButton("Load Memory (.ptmem)...")
            self.load_mem_btn.clicked.connect(self.on_load_memory)
            btn_row.addWidget(self.save_json_btn)
            btn_row.addWidget(self.save_mem_btn)
            btn_row.addWidget(self.load_mem_btn)
            right_layout.addLayout(btn_row)
            splitter.addWidget(right)

            splitter.setSizes([380, 700])
            layout.addWidget(splitter)
            return w

        def _build_recall_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            controls = QGroupBox("Content-addressable recall (Task A, Eq. 1 / 11)")
            form = QFormLayout(controls)

            self.recall_combo_list = QListWidget()
            self.recall_combo_list.setMinimumHeight(150)
            self.recall_combo_list.setMaximumHeight(220)
            form.addRow(QLabel("Pick a stored chunk to recall:"), self.recall_combo_list)

            noise_row = QHBoxLayout()
            self.noise_slider = QSlider(Qt.Horizontal)
            self.noise_slider.setRange(0, 90)
            self.noise_slider.setValue(25)
            self.noise_label = QLabel("25% bits flipped")
            self.noise_slider.valueChanged.connect(
                lambda v: self.noise_label.setText(f"{v}% bits flipped")
            )
            noise_row.addWidget(self.noise_slider)
            noise_row.addWidget(self.noise_label)
            form.addRow(QLabel("Cue noise:"), noise_row)

            self.recall_steps_spin = QSpinBox()
            self.recall_steps_spin.setRange(5, 300)
            self.recall_steps_spin.setValue(60)
            form.addRow(QLabel("Max steps:"), self.recall_steps_spin)

            self.recall_btn = QPushButton("Run Recall")
            self.recall_btn.clicked.connect(self.on_run_recall)
            form.addRow(self.recall_btn)

            layout.addWidget(controls)

            self.recall_result_label = QLabel("Build a memory, then run a recall.")
            self.recall_result_label.setWordWrap(True)
            layout.addWidget(self.recall_result_label)

            self.recall_canvas = MplCanvas(nrows=1, ncols=2, width=9, height=3.4)
            layout.addWidget(self.recall_canvas)
            return w

        def _build_sequence_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            controls = QGroupBox("Gated sequence replay (Task B, Eq. 11 / 12)")
            form = QFormLayout(controls)
            self.replay_steps_spin = QSpinBox()
            self.replay_steps_spin.setRange(5, 2000)
            self.replay_steps_spin.setValue(60)
            self.replay_steps_spin.setToolTip("At one chunk per gate cycle a sequence of k chunks needs about 12*k steps.")
            form.addRow(QLabel("Steps:"), self.replay_steps_spin)
            self.theta_spin = QDoubleSpinBox()
            self.theta_spin.setRange(-1.0, 0.99)
            self.theta_spin.setSingleStep(0.05)
            self.theta_spin.setDecimals(2)
            self.theta_spin.setValue(0.0)
            self.theta_spin.setToolTip("Gate threshold theta_g in Eq. 12. 0 = paper default (gate open about half of each cycle). "
                                       "Higher = narrower gate; ~0.9 gives one chunk per cycle.")
            form.addRow(QLabel("Gate threshold \u03b8_g (higher = narrower):"), self.theta_spin)
            self.replay_btn = QPushButton("Replay Sequence")
            self.replay_btn.clicked.connect(self.on_run_replay)
            form.addRow(self.replay_btn)
            layout.addWidget(controls)

            self.replay_canvas = MplCanvas(nrows=2, ncols=1, width=9, height=5.2)
            self.replay_canvas.setMinimumHeight(320)
            layout.addWidget(self.replay_canvas, 1)

            self.replay_score_label = QLabel("")
            self.replay_score_label.setWordWrap(True)
            layout.addWidget(self.replay_score_label)
            layout.addWidget(QLabel("Retrieved order (best-matching chunk whenever it changes):"))
            self.replay_text = QPlainTextEdit()
            self.replay_text.setReadOnly(True)
            self.replay_text.setFixedHeight(90)
            layout.addWidget(self.replay_text)
            return w

        def _build_diagnostics_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)
            self.diag_warn = QLabel("")
            self.diag_warn.setWordWrap(True)
            self.diag_warn.setStyleSheet("color: #ff6b5e; font-weight: 600;")
            self.diag_warn.hide()
            layout.addWidget(self.diag_warn)
            self.diag_label = QLabel("Build a memory to see capacity and stability diagnostics.")
            self.diag_label.setWordWrap(True)
            layout.addWidget(self.diag_label)
            self.diag_canvas = MplCanvas(nrows=1, ncols=2, width=11, height=4.8)
            layout.addWidget(self.diag_canvas)
            return w

        # ---------------------------------------------------------- logging
        def _log(self, msg: str):
            self.log.appendPlainText(msg)
            self.status.showMessage(msg, 5000)

        # ------------------------------------------------------- build flow
        def on_build_clicked(self):
            url = self.url_edit.text().strip()
            if not url:
                QMessageBox.warning(self, "Missing URL", "Please enter a URL first.")
                return
            self.build_btn.setEnabled(False)
            self.log.clear()
            self._log(f"Starting build for {url} ...")

            self.worker = BuildWorker(
                url, self.n_spin.value(), self.seq_check.isChecked(),
                rule=self.rule_combo.currentData(), auto_n=self.auto_n_check.isChecked(),
                decorrelate=self.decor_check.isChecked(), clean=self.clean_check.isChecked(),
                alpha=self.alpha_spin.value(),
            )
            self.worker.progress.connect(self._log)
            self.worker.finished_ok.connect(self.on_build_finished)
            self.worker.failed.connect(self.on_build_failed)
            self.worker.start()

        def on_build_failed(self, message: str):
            self.build_btn.setEnabled(True)
            self._log(f"ERROR: {message}")
            QMessageBox.critical(self, "Build failed", message)

        def _populate_chunk_lists(self):
            self.chunk_list.clear()
            self.recall_combo_list.clear()
            if self.record:
                for c in self.record["chunks"]:
                    item_text = f"#{c['id']:02d} [{c['tag']}] {c['text'][:90]}"
                    self.chunk_list.addItem(item_text)
                    self.recall_combo_list.addItem(item_text)
                self.json_view.setPlainText(dumps_record(self.record))
            elif self.network is not None:
                for i, lbl in enumerate(self.network.labels):
                    self.chunk_list.addItem(f"#{i:02d} {lbl}")
                    self.recall_combo_list.addItem(f"#{i:02d} {lbl}")
            if self.recall_combo_list.count():
                self.recall_combo_list.setCurrentRow(0)

        def on_build_finished(self, payload: Dict):
            self.build_btn.setEnabled(True)
            self.record = payload["record"]
            self.network = payload["network"]
            self.encoder = payload["encoder"]

            self.n_spin.setValue(self.network.N)      # reflect auto-sized N
            self.alpha_spin.setValue(self.network.alpha)
            self.network.theta_g = self.theta_spin.value()   # keep the saved memory consistent with the spin box
            self._populate_chunk_lists()
            self._log(f"Built memory with N={self.network.N} neurons and "
                      f"{len(self.network.patterns)} stored patterns.")
            warn = self.network.capacity_warning()
            if warn:
                self._log(f"WARNING: {warn}")
            self._refresh_diagnostics()

        # ------------------------------------------------------------ recall
        def on_run_recall(self):
            if self.network is None or self.encoder is None:
                QMessageBox.information(self, "No memory yet", "Build a memory first.")
                return
            row = self.recall_combo_list.currentRow()
            if row < 0:
                row = 0
            pattern = self.network.patterns[row].cpu().numpy()
            noise_frac = self.noise_slider.value() / 100.0
            rng = np.random.RandomState(row * 97 + 1)
            cue = BipolarEncoder.corrupt(pattern, noise_frac, rng=rng)

            steps = self.recall_steps_spin.value()
            result = self.network.recall(cue, steps=steps, gate_val=0.0)

            target_label = self.network.labels[row]
            got_idx = result["final_best_idx"]
            got_label = self.network.labels[got_idx] if got_idx >= 0 else "?"
            correct = (got_idx == row)
            conv = result["converged_at"]
            conv_str = f"step {conv}" if conv is not None else "did not fully converge in the given steps"
            if correct:
                verdict = "MATCH" if result["final_overlap"] >= 0.9 else "MATCH (degraded, overlap < 0.9)"
            else:
                verdict = ("MISMATCH (blended state, overlap < 0.9)" if result["final_overlap"] < 0.9
                           else "MISMATCH (settled in another stored chunk)")

            self.recall_result_label.setText(
                f"Cue = chunk #{row} with {int(noise_frac*100)}% of bits flipped.\n"
                f"Recalled: chunk #{got_idx} (overlap {result['final_overlap']:.3f}), "
                f"{verdict} vs. target chunk #{row}. Converged at {conv_str}.\n"
                f"Target: {target_label}\nRecalled: {got_label}"
            )

            ax_overlap, ax_energy = self.recall_canvas.axes
            ax_overlap.clear()
            ax_overlap.plot(result["overlap_trace"], color="#2f6fed")
            ax_overlap.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
            ax_overlap.set_title("Overlap with best-matching pattern")
            ax_overlap.set_xlabel("step t")
            ax_overlap.set_ylabel("overlap m(t)")
            ax_overlap.set_ylim(-1.05, 1.05)

            ax_energy.clear()
            ax_energy.plot(result["energy_trace"], color="#e0622b")
            ax_energy.set_title("Attractor energy H(x(t))  (Eq. context Sec 3.1)")
            ax_energy.set_xlabel("step t")
            ax_energy.set_ylabel("H(x)")

            self.recall_canvas.draw()

        # -------------------------------------------------------------- replay
        def on_run_replay(self):
            if self.network is None:
                QMessageBox.information(self, "No memory yet", "Build a memory first.")
                return
            steps = self.replay_steps_spin.value()
            self.network.theta_g = self.theta_spin.value()
            result = self.network.replay_sequence(steps=steps)
            score = self.network.score_replay(result["index_trace"])

            ax_gate, ax_idx = self.replay_canvas.axes
            ax_gate.clear()
            ax_gate.plot(result["gate_trace"], color="#8a3ffc")
            ax_gate.set_title("Gate g(t)  (Eq. 12)")
            ax_gate.set_ylabel("g(t)")

            ax_idx.clear()
            ax_idx.step(range(len(result["index_trace"])), result["index_trace"], where="mid", color="#0f9d58")
            ax_idx.set_title("Best-matching stored chunk over time")
            ax_idx.set_xlabel("step t")
            ax_idx.set_ylabel("chunk index")

            self.replay_canvas.fig.tight_layout()
            self.replay_canvas.draw()

            self.replay_score_label.setText(
                f"Replay order check: {score['correct']}/{score['transitions']} chunk changes went to the stored "
                f"successor; {score['in_order_prefix']} in a row before the first error; "
                f"{score['advances_per_cycle']:.2f} chunk changes per gate cycle"
                + (f"; whole sequence replayed in order by step {score['steps_to_end']}." if score["completed"] else "; the whole sequence was not replayed in order.")
            )
            lines = []
            prev = None
            for t, idx in enumerate(result["index_trace"]):
                if idx != prev:
                    label = self.network.labels[idx] if 0 <= idx < len(self.network.labels) else "?"
                    lines.append(f"t={t:3d}  ->  #{idx:02d}  {label}")
                    prev = idx
            self.replay_text.setPlainText("\n".join(lines))

        # ---------------------------------------------------------- diagnostics
        def _refresh_diagnostics(self):
            if self.network is None:
                return
            net = self.network
            rep = net.stability_report()
            caps = net.capacity_estimates()

            rec_txt = "   |   ".join(
                f"recall @ {int(round(float(k) * 100))}% noise: {v['accuracy'] * 100:.0f}% "
                f"({v['clean'] * 100:.0f}% clean)" for k, v in rep["recall"].items()
            )
            self.diag_label.setText(
                f"N = {net.N} neurons   |   P = {rep['P']} stored chunks (load P/N = {rep['load']:.2f})   |   "
                f"rule = {net.rule}   |   beta = {net.beta}   |   gamma = {net.gamma}\n"
                f"Estimated pattern capacity Pmax ~ 0.14N = {caps['Pmax_hebbian']:.1f}   |   "
                f"Estimated sequence capacity Smax ~ N/(c ln N) = "
                f"[{caps['Smax_low_c4']:.1f}, {caps['Smax_high_c2']:.1f}] for c in [4, 2]\n"
                f"Spectral radius rho(J) at stored chunk #0 = {rep['rho0']:.3f} "
                f"(should be < 1 for a stable attractor, Eq. 3-4)   |   "
                f"stable chunks: {rep['frac_stable'] * 100:.0f}% of {rep['P']}\n"
                f"Pattern crosstalk: mean |xi.xi'|/N = {rep['mean_abs_overlap']:.3f}, max = {rep['max_abs_overlap']:.3f}   |   {rec_txt}"
            )
            if rep["warning"]:
                self.diag_warn.setText("Warning: " + rep["warning"])
                self.diag_warn.show()
            else:
                self.diag_warn.hide()

            # Rebuild the figure each time so colour bars do not pile up across builds.
            fig = self.diag_canvas.fig
            fig.clear()
            ax_w, ax_c = fig.subplots(1, 2)
            W = net.W_A.cpu().numpy()
            lim = max(float(np.percentile(np.abs(W), 99)), 1e-6)
            im = ax_w.imshow(W, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
            ax_w.set_title("Attractor weight matrix W_A")
            ax_w.set_xlabel("neuron j")
            ax_w.set_ylabel("neuron i")
            fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)

            C = net.overlap_matrix()
            im2 = ax_c.imshow(C, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax_c.set_title("Pattern overlap xi_mu . xi_nu / N  (off-diagonal = crosstalk)")
            ax_c.set_xlabel("chunk nu")
            ax_c.set_ylabel("chunk mu")
            fig.colorbar(im2, ax=ax_c, fraction=0.046, pad=0.04)
            fig.tight_layout()
            self.diag_canvas.draw()

        # ------------------------------------------------------------- saving
        def on_save_json(self):
            if self.record is None:
                QMessageBox.information(self, "Nothing to save", "Build a memory first.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save JSON record", "memory_record.json", "JSON (*.json)")
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                f.write(dumps_record(self.record))
            self._log(f"Saved JSON record to {path}")

        def on_save_memory(self):
            if self.network is None:
                QMessageBox.information(self, "Nothing to save", "Build a memory first.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save memory", "memory.ptmem", "PyTorch memory (*.ptmem)")
            if not path:
                return
            payload = {"state": self.network.to_state(), "record": self.record}
            if self.encoder is not None:
                payload["encoder"] = self.encoder.to_state()
            torch.save(payload, path)
            self._log(f"Saved memory (weights + JSON record + encoder) to {path}")

        def on_load_memory(self):
            path, _ = QFileDialog.getOpenFileName(self, "Load memory", "", "PyTorch memory (*.ptmem)")
            if not path:
                return
            try:
                # weights_only=True: refuse arbitrary pickled objects (a .ptmem from someone else
                # must not be able to run code on load). Everything we save is tensors/lists/dicts/str/num.
                payload = torch.load(path, weights_only=True)
                self.network = DynamicMemoryNetwork.from_state(payload["state"])
                self.record = payload.get("record")
                enc_state = payload.get("encoder")
                self.encoder = (BipolarEncoder.from_state(enc_state) if enc_state
                                else BipolarEncoder(dim=self.network.N))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Load failed", str(exc))
                return

            self.n_spin.setValue(self.network.N)
            self.alpha_spin.setValue(self.network.alpha)
            self.theta_spin.setValue(self.network.theta_g)
            idx = self.rule_combo.findData(self.network.rule)
            if idx >= 0:
                self.rule_combo.setCurrentIndex(idx)
            self._populate_chunk_lists()
            self._log(f"Loaded memory from {path}")
            self._refresh_diagnostics()

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()


# ============================================================================
# 6. Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Memory as Dynamics -- URL memory encoder")
    parser.add_argument("--headless", metavar="URL", help="Run the pipeline on URL without the GUI (needs network).")
    parser.add_argument("--selftest", action="store_true", help="Run an offline pipeline check (no network, no GUI).")
    parser.add_argument("--benchmark", nargs="?", const="synthetic", metavar="URL",
                        help="Print a recall/stability/replay table for the paper setup vs. the extensions. "
                             "Without a URL a synthetic correlated corpus is used (offline).")
    parser.add_argument("--neurons", type=int, default=256, help="Number of neurons N (default: 256).")
    parser.add_argument("--rule", choices=["hebbian", "projection"], default="hebbian", help="Learning rule for --headless.")
    parser.add_argument("--decorrelate", action="store_true", help="IDF + centring encoder for --headless.")
    parser.add_argument("--auto-n", action="store_true", help="Auto-size N for --headless.")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if args.benchmark:
        if args.benchmark == "synthetic":
            run_benchmark(synthetic_corpus(), label="(synthetic correlated corpus, NOT a real page)")
        else:
            scraped = scrape_url(args.benchmark)
            run_benchmark([c["text"] for c in scraped["chunks"]], label=f"({args.benchmark})")
        return
    if args.headless:
        run_headless(args.headless, n_neurons=args.neurons, rule=args.rule,
                     decorrelate=args.decorrelate, auto_n=args.auto_n)
        return

    try:
        sys.exit(_launch_gui())
    except ImportError as exc:
        print("Could not start the GUI -- missing a Qt binding for QtPy.")
        print("Install one with e.g.  pip install qtpy PyQt5")
        print(f"Details: {exc}")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
